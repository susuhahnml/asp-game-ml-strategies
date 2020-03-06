#!/usr/bin/env python
# -*- coding: utf-8 -*-

from structures.action import Action, ActionExpanded
from py_utils.clingo_utils import *

class NodeMCTS:
    """
    A class used to represent a Node in a monte carlo tree search

    Attributes
    ----------
    state : State
        the state in which the action is taken
    action : Action
        the action taken in the step. For terminal states, action is None.
    time_step : int
        the time number in which the step was taken
    n : int
        The number of times it has been visited
    t : int
        The value calculated for node 
    """
    def __init__(self,state,action,time_step,n,t):
        self.state = state
        self.action = action
        self.time_step = time_step
        self.n = n
        self.t = t

    def __eq__(self, other):
        eq = True
        eq = self.state == other.state
        eq = eq and self.action == other.action
        eq = eq and self.n == other.n
        eq = eq and self.t == other.t
        return eq

    @classmethod
    def from_dic(cls,dic,game_def):
        """
        Constructs a Step from a dictionary
        """
        from structures.state import State
        t = dic['t']
        n = dic['n']
        time_step = dic['time_step']
        state = State.from_facts(dic['state'],game_def)
        action = None if dic['action'] is None else Action.from_facts(dic['action'],game_def)
        s = cls(state,action,time_step,n,t)
        return s

    def to_dic(self):
        """
        Returns a serializable dictionary to dump on a json
        """
        return {
            "t": self.t,
            "n": self.n,
            "time_step": self.time_step,
            "state": self.state.to_facts(),
            "action": None if self.action is None else self.action.to_facts()
        }

    def fluents_to_asp_syntax(self):
        """
        Returns all the fluents of the current state in asp syntax
        """
        return fluents_to_asp_syntax(self.state.fluents,self.time_step)

    def action_to_asp_syntax(self):
        """
        Returns the perfomed action in asp syntax
        """
        return action_to_asp_syntax(self.action,self.time_step)

    def to_asp_syntax(self):
        """
        Returns the state and action in asp syntax
        """
        fluents_str = self.fluents_to_asp_syntax()
        if(self.action):
            action_str = self.action_to_asp_syntax()
            fluents_str += action_str
        return fluents_str


    def __str__(self):
        """
        Returns a condensed string representation of the node
        """
        s=""
        if self.action:
            s= "t:{}, n:{}, {}".format(self.t,self.n,self.action)
        else:
            s= "t:{}, n:{}, {}".format(self.t,self.n,"No action")
        return s


    @property
    def ascii(self):
        """
        Returns the ascii representation of the step using the game definition
        """
        s = self.state.game_def.step_to_ascii(self)
        s ="〔t:{} n:{}〕\n{}".format(self.t,self.n,s)
        return s