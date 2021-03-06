#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from .action import Action
from .match import Match
from .step import Step
from py_utils.logger import log
import json
class NodeBase(Node):
    """
    Class representing a basic node of a game tree.
    Args:
        step: The step in the node with an state and the action performed
              in such state to take the game to this node
    """
    def __init__(self, step, main_player, dic={}, parent = None, children = [], name=None):
        super().__init__(name, parent=parent, children = children)
        self.step = step
        self.main_player = main_player

    @classmethod
    def from_dic(cls, dic, game_def, main_player, parent = None, children = []):
        """
        Crates a node from a dictionary, used for loading trees from a file
        """
        time_step = dic['time_step']
        state = State.from_facts(dic['step']['state'],game_def)
        action = None if dic['step']['action'] is None else Action.from_facts(dic['step']['action'],game_def)
        s = cls(Step(state, action, time_step),main_player, dic = dic, parent=parent, children=children)
        return s

    def to_dic(self):
        """
        Transforms the node into a dictionary for serialization in a json file
        """
        d = {"step": self.step.to_dic()}
        self.add_info_to_dic(d)
        return d

    def add_info_to_dic(self,dic):
        """
        Adds its additional information to a dic. Can be used by subclasses to add information.
        """
        pass

    def style(self):
        """
        Returns the style of the node for visualization
        """
        style = ['rounded','filled']
        if not (self.step.action is None):
            if self.step.action.player == self.main_player:
                style.append('solid')
            else:
                style.append('dotted')
        format_str = 'shape="box" style="%s"  fillcolor="#00000007" fontName="Bookman Old Style" labeljust=l'  % ( ",".join(style))
        return format_str

    @property
    def ascii(self):
        """
        Generate a label for visualization
        """
        if not self.step.action is None:
            return self.step.ascii
        else:
            if(self.step.state.is_terminal):
                return ("〔a:{} b:{}〕".format(self.step.state.goals['a'],self.step.state.goals['b']))
            else:
                other_player = "b" if self.main_player=="a" else "a"
                s ="〔INITIAL〕\n{}".format(self.step.ascii)
                return s


class Tree:
    """
    Tree class to handle search trees for games
    """
    node_class = NodeBase

    def __init__(self,root=None,main_player="a"):
        """ Initialize with empty root node and game class """
        self.root = root
        self.main_player = main_player
    
    @classmethod
    def load_from_file(cls, file_path, game_def):
        """
        Creates a Tree from a file with the tree in json format
        Args:
            file_path: Path to the json file
        """
        with open(file_path) as feedsjson:
            tree_dic = json.load(feedsjson)
        importer = DictImporter()
        root = importer.import_(tree_dic['tree'])
        for n in PreOrderIter(root):
            n = cls.node_class.from_dic(n.name,game_def,tree_dic['main_player'],parent=n.parent,children=n.children)
        t = cls(root)
        return t

    def create_node(self, step, *argv):
        """
        Creates a new node from the current class
        """
        return self.__class__.node_class(step,self.main_player,*argv) 

    def find_by_state(self, state):
        """
        Finds all nodes from tree matching a state
        """
        return findall(self.root, lambda n: n.step.state==state)

    def get_number_of_nodes(self):
        """
        Gets the number of nodes of the tree
        """
        nodes = findall(self.root)
        return len(nodes)
    
            
    def save_in_file(self,file_path):
        """
        Saves the tree in a json file
        """
        for n in PreOrderIter(self.root):
            n.name=n.to_dic()
        exporter = DictExporter()
        tree_json = exporter.export(self.root)
        final_json = {'main_player':self.main_player,'tree':tree_json}
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as feedsjson:
            json.dump(final_json, feedsjson, indent=4)

    def print_in_file(self,
                      file_name="tree_test.png"):
        """
        Function to plot generated tree as an image file

        Args:
            file_name (str): full name of image to be created
        """
        base_dir="./img/"
        image_file_name = base_dir + file_name
        # define local functions
        
        def aux(n):
            a = 'label="{}" {}'.format(n.ascii, n.style(parent=n.parent))
            return a
        # self.remove_leaves()
        os.makedirs(os.path.dirname(image_file_name), exist_ok=True)
        UniqueDotExporter(self.root,
                          nodeattrfunc=aux,
                          edgeattrfunc=lambda parent, child: 'arrowhead=vee').to_picture(image_file_name)
        log.info("Tree image saved in {}".format(image_file_name))

    @classmethod
    def node_from_match_initial(cls,match,main_player):
        """
        Function to construct a tree from a match class.
        It will create a tree with one branch representing the match
        Adds an extra root for the tree with the initial state.
        Args:
            match (Match): match to generate a tree

        Returns:
            root_node (anytree.Node): tree corresponding to match
        """
        root_node = cls.node_class(Step(match.steps[0].state,None,-1),main_player)
        rest = cls.node_from_match(match,main_player)
        rest.parent = root_node
        return root_node

    @classmethod
    def node_from_match(cls,match,main_player):
        """
        Function to construct a tree from a match class.
        It will create a tree with one branch representing the match
        Args:
            match (Match): a constructed match

        Returns:
            root_node (anytree.Node): tree corresponding to match
        """
        root_node = cls.node_class(match.steps[0],main_player=main_player,children=[])
        current_node = root_node
        for s in match.steps[1:]:
            new = cls.node_class(s,main_player=main_player,parent=current_node)
            current_node = new
        return root_node

    def remove_leaves(self):
        """
        Removes all leaves from the tree
        """
        leaves = self.root.leaves
        for l in leaves: l.parent = None 