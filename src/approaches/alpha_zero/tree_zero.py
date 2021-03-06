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
from structures.tree_MCTS import TreeMCTS,NodeMCTS
from py_utils.logger import log
import json
from structures.tree import Tree, NodeBase

class NodeZero(NodeMCTS):
    """
    Class representing a node from tree for the Alpha zero approach
    """
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
        """
        Gets a numpy array with the probabilities 
        for each legal action calculated in the tree
        and 0 in all illegal actins.
        """
        try:
            pi = np.zeros(game_def.encoder.action_size)
            for n in self.children:
                pi[game_def.encoder.actionstr_to_idx[str(n.step.action.action)]]=n.prob
            return pi/pi.sum()
        except Exception as e:
            raise(e)

class TreeZero(TreeMCTS):
    """
    Class representing tree for the Alpha zero approach
    """
    node_class = NodeZero
    def __init__(self,root,game_def,net,main_player="a"):
        """ Initialize with empty root node and game class """
        super().__init__(root,game_def,main_player)
        self.net = net

    def ucb1(self, node, expl, main_player, pi):
        """
        Function to calculate the best node to be visited
        """
        if node.step.state.is_terminal:
            return 1
        p_model = pi[self.game_def.encoder.actionstr_to_idx[str(node.step.action.action)]]
        if node.n == 0:
            return math.inf
        r = node.q_value 
        r += expl*p_model*(math.sqrt(node.parent.n)/(1+node.n))
        return r

    def rollout(self, node):
        """
        Unlike the rollout from MCTS a match is never simulated, the prediction of the network is used instead.
        """
        state = node.step.state
        p = state.control
        if state.is_terminal:
            return state.goals[p]
        pi, v = self.net.predict_state(node.step.state)
        return v

    def tree_traverse(self, node, expl):
        """
        Same function as the one in MCTS tree but calculating
        the networks prediction here to avoid multiple calculations
        """
        if node.is_leaf:
            if node.n == 0:
                next_node = node
            else:
                self.expand(node)
                if node.is_leaf:
                    next_node = node
                else:
                    next_node = node.children[0]
            v = self.rollout(next_node)
            self.backprop(next_node,v)
        else:
            pi, v = self.net.predict_state(node.children[0].step.state)
            next_node = max(node.children,key= lambda x:self.ucb1(x,expl,self.main_player,pi)) 
            self.tree_traverse(next_node,expl=expl)


    @staticmethod
    def run_episode(game_def, net, expl=0.3):
        """
        Runs one episode to generate examples.
        The full episode will run a MCTS simulation in the root
        if the tree, the choose the best action with the computed
        probabilities and add this as one example. The process will
        then be repeated with the new node until a terminal node in reached. It will generate as many examples as steps in the match.
        """
        examples = []
        state = game_def.get_initial_state()

        root = TreeZero.node_class(Step(state,None,0),"a")
        
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
            tree.run_mcts(net.args.n_mcts_simulations,expl=expl)
            # if j==1: tree.print_in_file("train-{}.png".format(j))
            if root.step.state.is_terminal:
                examples.append((root.step.state,[game_def.encoder.mask_state(current_state), np.zeros(game_def.encoder.action_size), None]))
                goals = root.step.state.goals
                v = goals[root.step.state.control]
                for s,e in examples[::-1]:
                    e[2]=v
                    # print("Example: \n{}\n{}\n".format(s,v)) 
                    v=-v
                return [e[1] for e in examples]

            pi = root.pis(game_def)
            
            examples.append((current_state,[game_def.encoder.mask_state(current_state), pi, None]))
            a = np.random.choice(game_def.encoder.all_actions, p=pi)
                
            root = [n for n in root.children if str(n.step.action.action)==str(a)][0]
