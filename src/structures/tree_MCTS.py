#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from .state import State, StateExpanded
from structures.action import Action
from .match import Match
from .step import Step
from py_utils.logger import log
import json
from structures.tree import Tree, NodeBase

class NodeMCTS(NodeBase):
    """
    Class representing a node from a tree tree for a Monte Carlo Tree search
    Args:
        step: The step in the node with an state and the action performed
              in such state to take the game to this node
    """
    def __init__(self, step, main_player, dic={}, parent = None, children = []):
        """
        Constructs a node
        Args:
            t: Value calculated with back prop. It is wrt the player in turn in the node
            n: Number of times it has been visited
        """
        super().__init__(step,main_player=main_player,parent=parent, children=children)
        self.t = 0 if not "t" in dic else dic["t"]
        self.n = 0 if not "n" in dic else dic["n"]

    def add_info_to_dic(self,dic):
        """
        Returns a serializable dictionary to dump on a json
        """
        dic["t"] = self.t
        dic["n"] = self.n

    def incremet_visits(self):
        """
        Increments the number of visits to the node
        """
        self.n = self.n + 1

    def incremet_value(self,t):
        """
        Increments the value of the node by t
        """
        self.t = self.t + t

    @property
    def prob(self):
        """
        Returns the probability of the node based on the parents visits
        """
        parent_n = self.n if self.parent is None else (self.parent.n-1)
        if(self.n==0):
            return 1
        return self.n/parent_n
    
    @property
    def q_value(self):
        """
        Returns the q value of the node. The average reward when this node 
        is visited
        """
        visited = self.n if self.n>0 else 1
        return self.t/visited
    
        
    @property
    def ascii(self):
        """
        Returns the ascii representation of the step including the visits and value
        Used for printing
        """
        
        p = round(self.prob,2)
        q = round(self.q_value,2)
        if not self.step.action is None:
            return "〔t:{} n:{} p:{} q:{}〕\n{}".format(self.t,self.n,p,q, self.step.ascii)
        else:
            if(self.step.state.is_terminal):
                return ("〔t:{} n:{} p:{} q:{}〕".format(self.t,self.n,p,q))
            else:
                other_player = "b" if self.main_player=="a" else "a"
                s ="〔t:{} n:{} p:{} q:{}〕\nmax:{}\nmin:{}\n{}".format(self.t,self.n,p,q,self.main_player,other_player,self.step.ascii)
                return s


    def style(self,parent):
        """
        Returns the style of the node for visualization
        """
        format_str = NodeBase.style(self)
        if parent is None:
            return format_str
        base = ' fillcolor="#466BCB{}"'
        a = self.p
        if a>0.98:
            base = base[:-3]+'"'
        else:
            base = base.format(a*100)
        
        format_str += base
        return format_str

    def __str__(self):
        return self.ascii

class TreeMCTS(Tree):
    """
    Class representing a tree for a Monte Carlo Tree search
    """
    node_class = NodeMCTS
    def __init__(self,root,game_def,main_player="a"):
        """ 
        Initialize with empty root node and game class 
        """
        super().__init__(root,main_player)
        self.game_def = game_def
        self.pa = RandomPlayer(game_def,"random",main_player)
        self.pb = RandomPlayer(game_def,"random",main_player)

    def get_best_action(self,node):
        """
        Gets the best action for a given node
        """
        next_n =  max(node.children, key=lambda n: n.n)
        return next_n.step.action
        
    def get_train_list(self):
        """
        Gets a list for generating training data
        """
        dic = {}
        for n in self.root.children:
            self.add_to_training_dic(dic,n)
        return dic.values()

    def add_to_training_dic(self,dic,node):
        """
        Adds the information from a node to a dictionary
        """
        if node.step in dic:
            log.info("Duplicated step")
            log.info(node.step)
            if dic[node.step]['n']>=node.n:
                return
        next_nodes = node.children
        if len(next_nodes) == 0:
            return
        dic[node.step] = {'s_init':node.step.state,
            'action':node.step.action,
            's_next':next_nodes[0].step.state,
            'p':node.prob,
            'n':node.n}
        
        for n in next_nodes:
            self.add_to_training_dic(dic,n)
        
    def save_values_in_file(self,file_path):
        """
        Saves the tree states as a dictionary to define best scores
        """
        state_dic = {}
        for n in PreOrderIter(self.root):
            if n.step.action is None:
                continue
            state_facts = n.step.state.to_facts()
            if not state_facts in state_dic:
                state_dic[state_facts] = {}
            state_dic[state_facts][n.step.action.to_facts()] = {'t':n.t,'n':n.n}

        final_json = {'main_player':self.main_player,'tree_values':state_dic}
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as feedsjson:
            json.dump(final_json, feedsjson, indent=4)   

    ################## Standar functions for MCTS 

    def run_mcts(self, n_iter, initial_node = None, expl=3 , check_early_stop=20):
        """
        Runs a MCTS

        Args:
            n_iter: Number of iterations to run the simulation
            inital_node: The initial node used as root
            expl: The exploration factor for the calculation of ubc1
            check_early_stop: After this number if iterations 
                              it will check if the tree has been updated
                              If the tree stays the same will reach an early stop.
        """
        node = self.root if initial_node is None else initial_node
        current_state = node.step.state
        for a in current_state.legal_actions:
            step = Step(current_state,a,node.step.time_step)
            self.__class__.node_class(step,self.main_player,parent=node)
        
        old_q = np.array([])
        for i in range(n_iter):
            self.tree_traverse(self.root,expl)
            if i%20==0:
                new_q = np.array([n.q_value for n in self.root.leaves])
                if np.array_equal(new_q,old_q):
                    break
                old_q=new_q

    def ucb1(self, node, expl):
        """
        Function to calculate the likelyhood of a node to be visited
        """
        if node.n == 0:
            return math.inf
        r = node.q_value 
        r += expl*(math.sqrt(node.parent.n/(1+node.n)))
        return r

    def tree_traverse(self, node, expl):
        """
        Transvers the tree from the given node.
        """
        if node.is_leaf:
            if node.n == 0:
                next_node = node
            else:
                self.expand(node)
                if node.is_leaf:
                    #Terminal state
                    next_node = node
                else:
                    #Go te first node
                    next_node = node.children[0]
            v = self.rollout(next_node)
            self.backprop(next_node,v)
        else:
            next_node = max(node.children,key= lambda x:self.ucb1(x,expl)) 
            self.tree_traverse(next_node,expl=expl)
    
    def expand(self, node):
        """
        Expands the node with the possible children given the legal actions
        """
        #Add one child per legal action
        if node.step.state.is_terminal:
            return
        current_action = node.step.action
        current_state = node.step.state.get_next(current_action)
        for a in current_state.legal_actions:
            step =Step(current_state,a,node.step.time_step)
            self.__class__.node_class(step,self.main_player,parent=node)
        if current_state.is_terminal:
            step =Step(current_state,None,node.step.time_step)
            self.__class__.node_class(step,self.main_player,parent=node)
         
    def rollout(self, node):
        """
        Makes a rollout, if the node is terminal returns the goals if not, 
        it simulates a match and returns the reached goals.
        """
        state = node.step.state
        p = state.control
        if state.is_terminal:
            return state.goals[p]
        state = state.get_next(node.step.action)
        self.game_def.initial = state.to_facts()
        match, benchmarks = Match.simulate(self.game_def,[self.pa,self.pb])
        return match.goals[p]

    def backprop(self, node, v):
        """
        Propagates the value up the tree starting from the node.
        The value will be switch symbol to always represent the
        reward from the point of view of the player in turn in the
        each node.
        """
        while(not node is None):
            node.incremet_visits()
            node.incremet_value(v)
            node = node.parent
            v=-v
        pass



    