from approaches.rl_agent.asp_game_env import ASPGameEnv
from approaches.rl_agent.tracker_callbacks import SaveTrackEpisodes
from approaches.rl_agent.model import ModelSelector
from approaches.rl_agent.agent import AgentSelector
from approaches.rl_agent.policy import PolicySelector
from approaches.strategy.player import StrategyPlayer
from approaches.random.player import RandomPlayer
from structures.game_def import GameDef
from py_utils.logger import log
from structures.players import Player
import numpy as np
import asyncio
import json

import tensorflow as tf
tf.compat.v1.disable_eager_execution()


class RLInstance():

	def __init__(self, architecture, agent, policy, epsilon, rewardf, opponent_name, n_steps, model_name, game_name, strategy_path):
		self.input = locals()
		self.game_name = game_name
		game_def = GameDef.from_name(game_name)
		self.build(architecture, agent, policy, epsilon, rewardf,opponent_name,n_steps, model_name, game_def, strategy_path)
		del self.input['self']
		
		
	def build(self, architecture, agent, policy, epsilon, rewardf, opponent_name, n_steps, model_name, game_def, strategy_path):
		

		opponent = Player.from_name_style(game_def,opponent_name,'a')
		clip_rewards = False
		if rewardf == "clipped":
			clip_rewards = True


		self.env = ASPGameEnv(game_def,opponent,clip_rewards= clip_rewards)
		self.model = ModelSelector(self.env.nb_observations, self.env.nb_actions).return_model(architecture)
		self.policy = PolicySelector(env = self.env, epsilon = epsilon).return_policy(policy)
		self.agent = AgentSelector(env = self.env, model = self.model, policy = self.policy).return_agent(agent)
		self.instance_name = model_name
		self.loop = asyncio.get_event_loop()
	

	def train(self, saving = True, num_steps = 1000):
		#np.random.seed(666)
		# env.seed(666)

		training_logger = SaveTrackEpisodes(name=self.instance_name,game_name=self.game_name)
		
		self.agent.fit(self.env, nb_steps=num_steps, visualize=False, nb_max_episode_steps=99,
			callbacks=[training_logger])

		if saving:
			self.save()	    

	def test(self, num_episodes = 2, visualize = True, nb_max_episode_steps=99):
		log.info("\n\nTesting ---------------\n")
		# recording test results
		self.agent.test(self.env, nb_episodes=num_episodes, visualize=visualize, nb_max_episode_steps=nb_max_episode_steps)
		self.env.game.debug = False

	def save(self):
		file_base = "./approaches/rl_agent/saved_models/{}/{}".format(self.game_name,self.instance_name)
		file_weights = file_base + ".weights"
		file_info = file_base + ".json"
		self.agent.save_weights(file_weights, overwrite=True)
		with open(file_info, 'w') as fp:
			json.dump(self.input, fp)
		log.info("RL instance saved in {}".format(file_base))


	def load_weights(self, model_name):
		file_name = "./approaches/rl_agent/saved_models/{}/{}.weights".format(self.game_name, model_name)
		self.agent.load_weights(file_name)


	@classmethod
	def from_file(cls, game_name, model_name):
		file_info = "./approaches/rl_agent/saved_models/{}/{}.json".format(game_name, model_name)
		with open(file_info, 'r') as fr:
			dct = json.load(fr)
		rl_instance = cls(**dct)
		rl_instance.load_weights(model_name)
		return rl_instance

