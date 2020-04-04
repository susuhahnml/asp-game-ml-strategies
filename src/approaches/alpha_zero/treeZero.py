#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
        if not self.step.action is None:
            return "〔p:{} q:{}〕\n{}".format(p,q, self.step.ascii)
        else:
            if(self.step.state.is_terminal):
                return ("〔p:{} q:{}〕".format(p,q))
            else:
                other_player = "b" if self.main_player=="a" else "a"
                s ="〔p:{} q:{}〕\nmax:{}\nmin:{}\n{}".format(p,q,self.main_player,other_player,self.step.ascii)
                return s


class TreeZero(TreeMCTS):
    """
    Tree class to handle search trees for games
    """
    node_class = NodeZero
    def __init__(self,root,game_def,main_player="a",model=None):
        """ Initialize with empty root node and game class """
        super().__init__(root,game_def,main_player)
        self.model = model

    def ucb1(self, node, expl, main_player):
        state_masked = self.game_def.encoder[main_player].mask_state(node.step.state)
        pi, v = self.model.predict(state_masked)
        p_model = pi[self.game_def.encoder.actionstr_to_idx[node.step.action.action]]
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
        state_masked = self.game_def.encoder[self.main_player].mask_state(node.step.state)
        pi, v = self.model.predict(state_masked)
        return v