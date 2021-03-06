from gym import Env
import numpy as np
from gym.spaces import Discrete, Tuple
from py_utils.colors import *
import sys
import copy
from approaches.dqn_rl.game_state import GameState
from approaches.random.player import RandomPlayer
from approaches.strategy.player import StrategyPlayer
from py_utils.logger import log
class GymEnv(Env):
    """The abstract environment class that is used by all agents. This class has the exact
    same API that OpenAI Gym uses so that integrating with it is trivial. In contrast to the
    OpenAI Gym implementation, this class only defines the abstract methods without any actual
    implementation.
    To implement your own environment, you need to define the following methods:
    - `step`
    - `reset` 
    - `render`
    - `close`
    Refer to the [Gym documentation](https://gym.openai.com/docs/#environments).
    """

    def __init__(self,game_def,possible_initial_states, player_name="a",opponent=None):
        self.game_def = game_def
        self.game_state= GameState(game_def,possible_initial_states)
        self.action_space = Discrete(self.game_def.encoder.action_size)
        self.observation_space = Tuple([Discrete(2) for i in range(0,self.game_def.encoder.state_size)])
        self.reward_range = (-100, 100)
        if opponent is None:
            log.info("Using random player as opponent")
            self.opponent = RandomPlayer(game_def,"","b")
        else:
            log.info("Loading strategy player as opponent from "+opponent)
            self.opponent = StrategyPlayer(game_def,"startegy-"+opponent,"b")



    def step(self, action):
        """Run one timestep of the environment's dynamics.  
       
        FOR BOTH PLAYERS!!!!!!

        Accepts an action and returns a tuple (observation, reward, done, info).
        # Arguments
            action (object): An action provided by the environment.
        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        #TODO what if not starting with a?
        #Performing one step for player a
        obs1, rew1, done1, info =  self.game_state.step("a",action,next_strategy =  self.opponent.strategy)
        if done1:
            return obs1, rew1, done1, info
        
        action = self.opponent.choose_action(self.game_state.current_state)
        action_idx = self.game_def.encoder.actionstr_to_idx[str(action.action)]
        # rand_idx = self.game.sample_random_legal_action()
        #Performing one random step for player b
        return self.game_state.step("b",action_idx)

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        #TODO find how to deep copy the game without the clingo errors
        self.game_state.random_reset()
        return self.game_state.current_observation

    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)
        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        self.game_state.render()

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        return

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        # TODO
        pass

    def configure(self, *args, **kwargs):
        """Provides runtime configuration to the environment.
        This configuration should consist of data that tells your
        environment how to run (such as an address of a remote server,
        or path to your ImageNet data). It should not affect the
        semantics of the environment.
        """
        # TODO
        pass

    def __del__(self):
        self.close()

    def __str__(self):
        return """
        Current state: {}
        """.format(self.game_state.current_state)
