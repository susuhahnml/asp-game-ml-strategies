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
    def __init__(self, step, main_player, dic={}, parent = None, children = []):
        """
        Constructs a node
        Args:
            t: Value calculated with back prop
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
        self.n = self.n + 1

    def incremet_value(self,t):
        self.t = self.t + t

    @property
    def prob(self):
        parent_n = self.n if self.parent is None else (self.parent.n-1)
        if(self.n==0):
            return 1
        return self.n/parent_n
    
    @property
    def q_value(self):
        visited = self.n if self.n>0 else 1
        return self.t/visited
    
    @property
    def is_almost_terminal(self):
        return len(self.children)==1 and self.children[0].step.action is None
        
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
        format_str = NodeBase.style(self)
        if parent is None:
            return format_str
        # a = self.q_value
        # base = ' fillcolor="#00FF00{}"' if a>0 else ' fillcolor="#FF0000{}"'
        # final = 0.8*(-a) if a<0 else (a)*0.2
        a = self.prob
        base = ' fillcolor="#00FF00{}"' if a>0.5 else ' fillcolor="#FF0000{}"'
        final = 0.8-a if a<0.5 else a -0.2
        
        alpha = "{0:0=2d}".format(int(final*100))
        format_str += base.format(alpha)
        return format_str

    def __str__(self):
        return self.ascii

class TreeMCTS(Tree):
    """
    Tree class to handle search trees for games
    """
    node_class = NodeMCTS
    def __init__(self,root,game_def,main_player="a"):
        """ Initialize with empty root node and game class """
        super().__init__(root,main_player)
        self.game_def = game_def
        self.pa = RandomPlayer(game_def,"random",main_player)
        self.pb = RandomPlayer(game_def,"random",main_player)

    def get_best_action(self,node):
        next_n =  max(node.children, key=lambda n: n.n)
        return next_n.step.action
        
    def get_train_list(self):
        dic = {}
        for n in self.root.children:
            self.add_to_training_dic(dic,n)
        return dic.values()

    def add_to_training_dic(self,dic,node):
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
        

    def run_mcts(self, n_iter, initial_node = None, expl=2 ):
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
                    log.debug("Early stopping MCTS in iteration {}".format(i))
                    break
                old_q=new_q

    def ucb1(self,node,expl,main_player):
        if node.n == 0:
            return math.inf
        r = node.q_value 
        if node.step.state.control != main_player:
            r = -1*r
        # r += expl*(math.sqrt(math.log(node.parent.n)/node.n))
        r += expl*(math.sqrt(node.parent.n/(1+node.n)))
        return r

    def tree_traverse(self, node, expl):
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
            next_node = max(node.children,key= lambda x:self.ucb1(x,expl,self.main_player)) 
            self.tree_traverse(next_node,expl=expl)
    
    def expand(self, node):
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
        state = node.step.state
        if state.is_terminal:
            return state.goals[self.main_player]
        state = state.get_next(node.step.action)
        self.game_def.initial = state.to_facts()
        match, benchmarks = Match.simulate(self.game_def,[self.pa,self.pb],signal_on =False)
        return match.goals[self.main_player]

    def backprop(self, node, v):
        while(not node is None):
            node.incremet_visits()
            node.incremet_value(v)
            node = node.parent
            # v=-v
        pass

    
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

    