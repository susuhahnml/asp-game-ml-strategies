from py_utils.colors import *
from collections import defaultdict
import numpy as np
from structures.state import StateExpanded, State
from structures.match import Match
from structures.step import Step
from py_utils.logger import log
class GameState:

    """
    Creates a an ongoing game from a game_def.
    Args: 
        game_def: Game Definition
    """
    def __init__(self, game_def,possible_initial_states):
        self.game_def = game_def
        self.possible_initial_states = possible_initial_states
        self.random_reset()

    @property
    def current_observation(self):
        return self.game_def.encoder.mask_state(self.current_state)
    

    """
    Gets a dictionary with the rewards for all players in the current state
    Returning 0 in by default
    """
    @property 
    def current_rewards(self):
        resdict = defaultdict(int,self.current_state.goals)
        return resdict

    """
    Resets the game with a random initial state
    """
    def random_reset(self):
        n_init = len(self.possible_initial_states)
        initial_idx = np.random.randint(n_init)
        self.game_def.initial = self.possible_initial_states[initial_idx]
        initial_state = self.game_def.get_initial_state()
        self.current_state = initial_state
        self.match = Match([Step(initial_state,None,0)])


    """Run one timestep of the games dynamics. !!!!FOR ONE PLAYER!!!!!
    Accepts an action index and returns a tuple (observation, reward, done, info).
    Args:
        action_idx (number): The index of the action performed my the player
        player_str: The player name from which we want the reward
    Returns
        observation (object): Agent's observation of the current environment. Hot-One List
        reward (float) : Amount of reward returned after selected action.
        done (boolean): Whether the episode has ended
        info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
    """
    def step(self, player_str, action_idx, next_strategy = None):
        log.debug(paint("\n----------- Performing GAME step -----------",bcolors.REF))
        action_str =  self.game_def.encoder.all_actions[action_idx]
        
        try:
            legal_action = self.current_state.get_legal_action_from_str(action_str)
            log.debug(legal_action)
        except RuntimeError as err: #Acounts for Ilegal Action
            log.debug(paint("\tSelected non legal action",bcolors.FAIL))
            player_reward = -100
            log.debug(paint_bool("••••••• EPISODE FINISHED Reward:{} •••••••".format(player_reward),player_reward>0))
            return self.current_observation, player_reward, True, {}
        #Construct next state
        self.match.add_step(Step(self.current_state,legal_action,len(self.match.steps)))
        next_state = self.current_state.get_next(legal_action, strategy_path=next_strategy)
        self.current_state = next_state
        
        #Get information from next state
        done = self.current_state.is_terminal
        goals_dic = self.current_rewards
        player_reward = goals_dic[player_str] 
        if(done):
            log.debug(paint_bool("••••••• EPISODE FINISHED Reward:{} •••••••".format(player_reward),player_reward>0))
        return self.current_observation, player_reward, done, {}

    def __str__(self):
        return self.match.steps[-1].ascii

    def render(self):
        last = self.match.steps[-1]
        if(last.time_step%2 == 0 and last.time_step>0):
            log.info("\n" + self.match.steps[-2].ascii + "\n\n" + self.match.steps[-1].ascii)
        else:
            log.info("\n" + self.match.steps[-1].ascii)
