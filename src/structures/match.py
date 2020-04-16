#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from random import randint
from py_utils.logger import log
from collections import defaultdict
from structures.state import State, StateExpanded
from structures.action import ActionExpanded, Action
from structures.players import IllegalActionError
from structures.step import Step
from py_utils.colors import *
import signal
import numpy as np
class Match:
    """
    Class to represent a match, this match is defined by a list of steps
    indicating the actions taken and their corresponding changes in the environment

    Attributes
    ----------
    steps : list(Step)
        list of steps performed in the match
    """
    def __init__(self,steps):
        self.steps = steps
        self.illegal_lost=None

    def add_step(self, step):
        """
        Adds a nex step to the list of steps
        Args:
            step (Step): Step to add in the end
        """
        self.steps.append(step)

    @property
    def goals(self):
        """
        Returns: Obtains the goals of a match using the last step
        """
        if(len(self.steps) == 0):
            return {}
        return self.steps[-1].state.goals

    def __str__(self):
        """
        Returns: A console representation of the match
        """
        s = ""
        c = [bcolors.OKBLUE,bcolors.HEADER]
        for step in self.steps:
            s+=c[step.time_step%2]
            s+="\nSTEP {}:\n".format(step.time_step)
            s+= step.ascii
            if(step.state.is_terminal):
                s+="\n{}GOALS: \n{}{}".format(bcolors.OKGREEN, step.state.goals,
                                            bcolors.ENDC)
            s+=bcolors.ENDC
        return s

    def generate_train(self, training_list, i):
        """
        Adds a training instance to the list for the decision made at time_step i
        Args:
            training_list ([Dic]): the list will all training instances
            i (int): the time step that must be added
        """
        if training_list is None:
            return
        step = self.steps[i]
        p = step.state.control
        control_goal = self.goals[p]
        training_list.append(
            {'s_init':step.state,
            'action':step.action,
            's_next':self.steps[i+1].state,
            'reward':control_goal,
            'win':-1 if control_goal<0 else 1})

    @staticmethod
    def simulate(game_def, players, depth=None, ran_init=False, signal_on=True,time_out_sec=3):
        """
        Call it with the path to the game definition

        Args:
            players (Player,Player): A tuple of the players

            depth:
                - n: Generate until depth n or terminal state reached
        """

        def handler(signum, frame):
            raise TimeoutError("Action time out")
        
        if signal_on: signal.signal(signal.SIGALRM, handler)
        if(ran_init):
            initial = game_def.get_random_initial()
        else:
            initial = game_def.initial
        state = StateExpanded.from_game_def(game_def,
                        initial,
                        strategy = players[0].strategy)
        match = Match([])
        time_step = 0
        continue_depth = True if depth==None else time_step<depth
        log.debug("\n--------------- Simulating match ----------------")
        log.debug("\na: {}\nb: {}\n".format(players[0].name,
                                                players[1].name))

        letters = ['a','b']
        response_times = {'a':[],'b':[]}
        while(not state.is_terminal and continue_depth):
            current_control= letters[time_step%2]
            if signal_on: signal.alarm(time_out_sec)
            t0 = time.time()
            try:
                selected_action = players[time_step%2].choose_action(state,time_step=time_step)
            except TimeoutError as ex:
                log.debug("Time out for player {}, choosing random action".format(current_control))
                index = randint(0,len(state.legal_actions)-1)
                selected_action = state.legal_actions[index]
            except IllegalActionError as ex:
                log.debug("Player {}, choosing illegal action {} in step {} -> Match lost".format(players[time_step%2].name,str(ex.action),time_step))
                state.is_terminal=True
                state.goals={current_control:-1,letters[(time_step+1)%2]:+1,}
                selected_action=None
                match.illegal_lost={"player":current_control,"time_step":time_step}
            if signal_on: signal.alarm(0)
            t1 = time.time()
            response_times[current_control].append(round((t1-t0)*1000,3))
            step = Step(state,selected_action,time_step)
            match.add_step(step)
            time_step+=1
            continue_depth = True if depth==None else time_step<depth
            if not selected_action is None:
                state = state.get_next(selected_action,
                                strategy_path = players[time_step%2].strategy)
        match.add_step(Step(state,None,time_step))
        log.debug(match)
        return match, {k:round(sum(lst) / (len(lst) if len(lst)>0 else 1),3) for k,lst in response_times.items()}

    @staticmethod
    def vs(game_def, n, player_encounters,initial_states,styles,signal_on=False,time_out_sec=3):
        scores = [
            {'wins':0,'draws':0,'points':0,'response_times':[],"matches_lost_by_illegal":0,"time_steps_lost_by_illegal":set()},
            {'wins':0,'draws':0,'points':0,'response_times':[],"matches_lost_by_illegal":0,
            "time_steps_lost_by_illegal":set()}]
        for i in range(n):
            for turn, vs in enumerate(player_encounters):
                idx = {'a':0+turn,'b':1-turn}
                game_def.initial = initial_states[i%len(initial_states)]
                match, metrics = Match.simulate(game_def,vs,ran_init=False,signal_on=signal_on,time_out_sec=time_out_sec)
                goals = match.goals
                for l,g in goals.items():
                    scores[idx[l]]['points']+=g
                    if g>0:
                        scores[idx[l]]['wins']+=1
                    elif g==0:
                        scores[idx[l]]['draws']+=1
                scores[idx['a']]['response_times'].append(metrics['a'])
                scores[idx['b']]['response_times'].append(metrics['b'])
                if not match.illegal_lost is None:
                    scores[idx[match.illegal_lost["player"]]]['matches_lost_by_illegal'] +=1
                    scores[idx[match.illegal_lost["player"]]]['time_steps_lost_by_illegal'].add(match.illegal_lost["time_step"])
        benchmarks = {}
        players = ['a','b']
        for i,p in enumerate(players):
            benchmarks[p]={} 
            benchmarks[p]['style_name']=styles[i]
            benchmarks[p]['wins']=scores[i]['wins']
            benchmarks[p]['wins']=scores[i]['wins']
            benchmarks[p]['total_reward']=scores[i]['points']
            response_times_np = np.array(scores[i]['response_times'])
            benchmarks[p]['average_response']=round(np.mean(response_times_np),3)
            benchmarks[p]['matches_lost_by_illegal']=scores[i]['matches_lost_by_illegal']
            benchmarks[p]['time_steps_lost_by_illegal']=list(scores[i]['time_steps_lost_by_illegal'])
            benchmarks[p]['std']=round(np.std(response_times_np),3)
        return benchmarks