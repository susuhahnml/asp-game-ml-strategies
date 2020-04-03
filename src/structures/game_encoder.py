from py_utils.clingo_utils import  generate_rule, get_all_possible
from py_utils.colors import *
from collections import defaultdict
import numpy as np
from structures.state import StateExpanded, State
from structures.match import Match
from structures.step import Step
from py_utils.logger import log
class GameEncoder:

    """
    Creates a game from a game_def.
    Args: 
        game_def: Game Definition
    """
    def __init__(self, game_def, clip_rewards = False, main_player="a"):
        self.game_def = game_def
        #Set all options for actions and observations
        all_actions, all_obs = get_all_possible(game_def, main_player)
        self.all_actions = all_actions
        self.actionstr_to_idx = {str(a):i for i,a in enumerate(all_actions)}
        self.all_obs = all_obs
        self.obsstr_to_idx = {str(o):i for i,o in enumerate(all_obs)}
        self.clip_rewards = clip_rewards
        #Set current state
        self.nb_actions = len(self.all_actions)
        self.nb_observations = len(self.all_obs)

    """
    Returns a nparray with True in the action and False in the rest
    """
    def mask_action(self,action):
        actions = np.zeros(len(self.all_actions))
        actions[self.actionstr_to_idx[action]] = 1
        return actions

    """
    Returns a nparray with True in the legal actions and False in the rest
    """
    def mask_legal_actions(self,state):
        actions = np.zeros(len(self.all_actions))
        legal_actions = state.get_symbol_legal_actions()
        for a in legal_actions:
            actions[self.actionstr_to_idx[a]] = 1
        return actions

    """
    The list defining the current obstervation with the size of all possible fluents.
    Contains 1 in the position of fluents that are true in the current state
    """
    @property
    def mask_state(self,state):
        obs = np.zeros(len(self.all_obs))
        fluents = state.fluents_str
        fluents = [f for f in fluents if f!='terminal']
        for f in fluents:
            obs[self.obsstr_to_idx[f]] = 1
        return obs

    """
    Randomly sample an action from leagal actions in current state.
    Returns the idx of the action.
    """
    def sample_random_legal_action(self, state):
        n_legal = len(state.legal_actions)
        if n_legal==0: # "Cant sample without legal actions"
            return None
        r_l_idx = np.random.randint(n_legal)
        legal_action_str = str(state.legal_actions[r_l_idx].action)
        real_idx = self.actionstr_to_idx[legal_action_str]

        return real_idx, legal_action_str

    """
    Gets a dictionary with the rewards for all players in the current state
    """
    @property 
    def current_rewards(self,state):
        resdict = defaultdict(int,state.goals)
        if self.clip_rewards:
            for key in resdict.keys():
                if resdict[key] > 0:
                    resdict[key] = 1
                elif resdict[key] < 0:
                    resdict[key] = -1
        return resdict