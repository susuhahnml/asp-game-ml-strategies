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

class NodeNet(NodeBase):
    def __init__(self, step, main_player, dic={}, parent = None, children = []):
        """
        Constructs a node
        Args:
            t: Value calculated with back prop
            n: Number of times it has been visited
        """
        super().__init__(step,main_player=main_player,dic=dic, parent=parent, children=children)
        self.is_legal = dic["is_legal"]
        self.p = dic["p"]
        self.v = dic["v"]

    @property
    def ascii(self):
        """
        Returns the ascii representation of the step including the visits and value
        Used for printing
        """
        p = str(round(self.p,2))
        v = str(round(self.v,2))
        if not self.step.action is None:
            return "〔p:{} v:{}〕\n{}".format(p,v, self.step.ascii)
        else:
            if(self.step.state.is_terminal):
                return ("〔p:{} v:{}〕".format(p,v))
            else:
                other_player = "b" if self.main_player=="a" else "a"
                s ="〔p:{} v:{}〕\nmax:{}\nmin:{}\n{}".format(p,v,self.main_player,other_player,self.step.ascii)
                return s

    def style(self,parent):
        format_str = NodeBase.style(self)
        if parent is None:
            return format_str
        # a = self.q_value
        # base = ' fillcolor="#00FF00{}"' if a>0 else ' fillcolor="#FF0000{}"'
        # final = 0.8*(-a) if a<0 else (a)*0.2
        if not self.is_legal:
            return format_str + ' fillcolor="#f5da42"'
        a = self.p
        base = ' fillcolor="#00FF00{}"' if a>0.5 else ' fillcolor="#FF0000{}"'
        final = 0.8-a if a<0.5 else a -0.2
        
        alpha = "{0:0=2d}".format(int(final*100))
        format_str += base.format(alpha)
        return format_str

class TreeNet(Tree):
    """
    Tree class to handle search trees for games
    """
    node_class = NodeNet
    def __init__(self,root,game_def,net,main_player="a"):
        """ Initialize with empty root node and game class """
        super().__init__(root,main_player)
        self.net = net
        self.game_def= game_def

    @classmethod
    def generate_from(cls,game_def,net,state,th=0.2):
        log.debug("Generating net tree...")
        root = TreeNet.node_class(Step(state,None,0),"a",dic={"is_legal":1,"p":1,"v":0})
        tree = TreeNet(root,game_def,net)
        current_nodes = [root]
        it = 0
        while(len(current_nodes)>0):
            it+=1
            new_nodes = []
        
            for n in current_nodes:
                s = n.step.state
                if s.is_terminal:
                    continue
                if not n.is_legal:
                    continue
                if n.step.action is None:
                    state = n.step.state
                else:
                    state = n.step.next_state()
                pi, v = net.predict_state(state)
                n.v=v
                legal_actions_masked = game_def.encoder.mask_legal_actions(state)
                for i,p in enumerate(pi):
                    if p<th and legal_actions_masked[i]==0:
                        continue
                    action_str= str(game_def.encoder.all_actions[i])
                    if legal_actions_masked[i]==0:
                        action = Action.from_facts("does({},{}).".format(state.control,action_str),game_def)
                    else:
                        action = state.get_legal_action_from_str(action_str) 
                    step = Step(state,action,n.step.time_step+1)
                    node = TreeNet.node_class(step,"a",parent=n,dic={"is_legal":legal_actions_masked[i]==1,"p":p,"v":0})
                    new_nodes.append(node)
            current_nodes = new_nodes
        return tree