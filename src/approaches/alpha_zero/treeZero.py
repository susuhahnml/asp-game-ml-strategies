#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import numpy as np
import math
from approaches.random.player import RandomPlayer
import os
import re
import anytree
from tqdm import tqdm
from anytree import Node, RenderTree, PreOrderIter
from anytree.exporter import UniqueDotExporter, DotExporter, DictExporter
from anytree.importer import DictImporter
from anytree.iterators.levelorderiter import LevelOrderIter
from anytree.search import findall
from structures.state import State, StateExpanded
from structures.action import Action
from structures.match import Match
from structures.step import Step
from structures.treeMCTS import TreeMCTS,NodeMCTS
from py_utils.logger import log
import json
from structures.tree import Tree, NodeBase
from approaches.alpha_zero.alpha_utils import predict

class NodeZero(NodeMCTS):
    def __init__(self, step, main_player, dic={}, parent = None, children = []):
        """
        Constructs a node
        Args:
            t: Value calculated with back prop
            n: Number of times it has been visited
        """
        super().__init__(step,main_player=main_player,dic=dic, parent=parent, children=children)

    @property
    def ascii(self):
        """
        Returns the ascii representation of the step including the visits and value
        Used for printing
        """
        
        p = round(self.prob,2)
        q = round(self.q_value,2)
        t = round(self.t,2)
        if not self.step.action is None:
            return "〔t:{} n:{} p:{} q:{}〕\n{}".format(t,self.n,p,q, self.step.ascii)
        else:
            if(self.step.state.is_terminal):
                return ("〔t:{} n:{} p:{} q:{}〕".format(t,self.n,p,q))
            else:
                other_player = "b" if self.main_player=="a" else "a"
                s ="〔t:{} n:{} p:{} q:{}〕\nmax:{}\nmin:{}\n{}".format(t,self.n,p,q,self.main_player,other_player,self.step.ascii)
                return s


    def pis(self,game_def):
        try:
            pi = np.zeros(game_def.encoder.action_size)
            for n in self.children:
                # print(str(n.step.action.action))
                pi[game_def.encoder.actionstr_to_idx[str(n.step.action.action)]]=n.prob
            # print(game_def.encoder.actionstr_to_idx)
            # print(pi)
            return pi/pi.sum()
        except Exception as e:
            raise(e)

class TreeZero(TreeMCTS):
    """
    Tree class to handle search trees for games
    """
    node_class = NodeZero
    def __init__(self,root,game_def,model,main_player="a"):
        """ Initialize with empty root node and game class """
        super().__init__(root,game_def,main_player)
        self.model = model

    def ucb1(self, node, expl, main_player):
        pi, v = predict(self.game_def,node.step.state,self.model)
        if node.step.state.is_terminal:
            return 1
        p_model = pi[self.game_def.encoder.actionstr_to_idx[str(node.step.action.action)]]
        if node.n == 0:
            return math.inf
        r = node.q_value 
        if node.step.state.control != main_player:
            r = -1*r
        r += expl*p_model*(math.sqrt(node.parent.n/(1+node.n)))
        return r

    def rollout(self, node):
        state = node.step.state
        if state.is_terminal:
            return state.goals[self.main_player]
        pi, v = predict(self.game_def,node.step.state,self.model)
        return v

    @staticmethod
    def run_episode(game_def, net, args,i):
        examples = []
        state = game_def.get_initial_state()

        root = TreeZero.node_class(Step(state,None,0),"a")
        # tree = TreeZero(root,game_def,net)
        
        current_state = state
        is_first = True
        j=0
        while True:
            j+=1
            if not is_first:
                current_state = root.step.next_state()
            else:
                current_state = state
                is_first = False
            root = TreeZero.node_class(Step(current_state,None,0),"a")
            tree = TreeZero(root,game_def,net)
            tree.run_mcts(args.n_mcts_simulations,expl=3)
            # tree.print_in_file("train-{}.png".format(j))
            if root.is_almost_terminal or root.step.state.is_terminal:
                if root.is_almost_terminal:
                    goals = root.children[0].step.state.goals
                else: 
                    goals = root.step.state.goals
                v = goals[root.step.state.control]

                for e in examples[::-1]:
                    e[2]=v
                    v=-v
                return examples

            pi = root.pis(game_def)
            # print("Example: \n{}\n{}".format(root.ascii,pi)) 
            # print(" ".join([str(n.step.action)+str(n.prob) for n in root.children]))
            
            # tree.print_in_file("train-{}-{}.png".format(i,time.time()))
            examples.append([game_def.encoder.mask_state(current_state), pi, None])
            try:
                a = np.random.choice(game_def.encoder.all_actions, p=pi)
            except Exception as e:
                tree.print_in_file("error-ch-{}.png".format(i))
                raise e
                
            root = [n for n in root.children if str(n.step.action.action)==str(a)][0]
