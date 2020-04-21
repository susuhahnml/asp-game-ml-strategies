from py_utils.clingo_utils import  generate_rule, has_player_ref
from py_utils.colors import *
from collections import defaultdict
import numpy as np
from structures.state import StateExpanded, State
from structures.match import Match
from structures.step import Step
from py_utils.logger import log

class GameEncoder:
    """
    A class used to encode a game definition and its states and actions
    into valued vectors using a hot-one encoding
    """
    def __init__(self, all_actions, all_obs):
        """
        Creates a game encoder from a game_def.
        Args: 
            all_actions (list): A list will all the possible actions 
            all_obs (list): A list with all possible observations (fluents)

            Automatically computed attributes:
            actionstr_to_idx (dic): A dictionary that given a string representation 
                                    of an action will return its index in the vector
            all_obs (dic): One element is created for each player ordering the
                           observations based on the player to make representations
                           equivalent across players
            obsstr_to_idx (dic): A dictionary that returns the index of an observation
            
            action_size (int): Number of actions
            state_size (int): Number of states (observations)

        """
        #Set all options for actions and observations
        all_actions = [str(a) for a in all_actions]
        self.all_actions = sorted(all_actions)
        self.actionstr_to_idx = {str(a):i for i,a in enumerate(self.all_actions)}
        self.all_obs = {
            "a": sorted(all_obs, key=lambda x: not has_player_ref(x,"a")),
            "b": sorted(all_obs, key=lambda x: not has_player_ref(x,"b"))
        }
        self.obsstr_to_idx = {
            "a": {str(o):i for i,o in enumerate(self.all_obs["a"])},
            "b": {str(o):i for i,o in enumerate(self.all_obs["b"])}
        }
        #Set current state
        self.action_size = len(self.all_actions)
        self.state_size = len(all_obs)

    def mask_action(self,action):
        """
        Returns a nparray with 1 in the action index and 0 in the rest

        Args:
            action (Action): The action to be masked
        """
        actions = np.zeros(len(self.all_actions))
        actions[self.actionstr_to_idx[action]] = 1
        return actions

    def mask_legal_actions(self,state):
        """
        Returns a nparray with 1 in the legal actions and 0 in the rest
        Args:
            state (StateExpanded): The state with the legal actions
        """
        actions = np.zeros(self.action_size)
        legal_actions = state.get_symbol_legal_actions()
        for a in legal_actions:
            actions[self.actionstr_to_idx[a]] = 1
        return actions

    def mask_state(self,state,main_player=None):
        """
        The list defining the current observation with the size of all possible fluents.
        Contains 1 in the position of fluents that are true in the given state
        Args:
            state (StateExpanded): The state to be masked
        """
        main_player = state.control if main_player is None else main_player
        obs = np.zeros(self.state_size)
        fluents = state.fluents_str
        fluents = [f for f in fluents if f!='terminal']
        for f in fluents:
            obs[self.obsstr_to_idx[main_player][f]] = 1
        return obs.astype(int)

    def sample_random_legal_action(self, state):
        """
        Choose one random legal action
        """
        n_legal = len(state.legal_actions)
        if n_legal==0: # "Cant sample without legal actions"
            return None
        r_l_idx = np.random.randint(n_legal)
        legal_action_str = str(state.legal_actions[r_l_idx].action)
        real_idx = self.actionstr_to_idx[legal_action_str]

        return real_idx, legal_action_str
