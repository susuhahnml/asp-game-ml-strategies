from sklearn.model_selection import cross_validate, train_test_split, cross_val_score
from approaches.rl_agent.asp_game_env import ASPGameEnv
from approaches.rl_agent.model import ModelSelector
from py_utils.logger import log
import numpy as np
import asyncio
import json
from approaches.rl_agent.game import Game
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential, Model


class MLInstance():

	def __init__(self, game_def, architecture, n_steps, model_name, training_file):
		self.input = locals()
		del self.input['self']
		del self.input['game_def']
		self.build(architecture, n_steps, model_name, game_def, training_file)
			
	def build(self, architecture, n_steps, model_name, game_def, training_file):
		self.game_def = game_def
		self.game = Game(game_def)
		self.instance_name = model_name
		self.training_file = training_file
		self.architecture = architecture
		n_a = self.game.nb_actions
		n_o = self.game.nb_observations
		self.input_size = n_a + n_o

		# self.loop = asyncio.get_event_loop()

	def divide_train_data(self):
		training_path = './approaches/ml_agent/train/{}/{}'.format(self.game_def.name, self.training_file)
		pandas_train = pd.read_csv(training_path,sep=';')
		pandas_train = pandas_train.sample(frac=1).reset_index(drop=True)
		train, test = train_test_split(pandas_train,test_size = 0.1)
		train_inputs = train.iloc[:,0:self.input_size]
		train_next_label = train.iloc[:,self.input_size:self.input_size+self.game.nb_observations]
		train_pred_label = train.iloc[:,-2:]
		test_inputs = test.iloc[:,0:self.input_size]
		test_next_label = test.iloc[:,self.input_size:self.input_size+self.game.nb_observations]
		test_pred_label = test.iloc[:,-2:]
		return (train_inputs,train_next_label,train_pred_label), (test_inputs,test_next_label,test_pred_label)


	
	def train(self, saving = True, num_steps = 1000):
		train, test = self.divide_train_data()
		print("Input")
		print(self.input_size)

		print(self.game.nb_observations)
		model_next = ModelSelector(self.input_size,self.game.nb_observations).return_model(self.architecture)

		model_next.compile(loss='categorical_crossentropy',
                    optimizer='adam', # Root Mean Square Propagation
                    metrics=['acc']) # Accuracy performance metric

		neural_network = KerasClassifier(build_fn=lambda: model_next, 
                                 epochs=10, 
                                 batch_size=100, 
                                 verbose=0)
		print(train[0].shape)
		score =  cross_validate(neural_network, train[0], train[1], cv=3)
		results = cross_val_score(neural_network, train[0], train[1], cv=3)
		
		print(results)
		neural_network.evaluate()


		

		#Partition data for cv

		#Use initial state and action state to train a network to learn the next state
		#Maybe save partial network for testing

		#Remove last layer of network and retrain to learn the reward and win
		#Save final network








	def test(self):
		#Test
		pass

	def save(self):
		file_base = "./ml_agent/saved_models/"+self
		file_weights = file_base + ".weights"
		file_info = file_base + ".json"
		#Save wigths of nn
		with open(file_info, 'w') as fp:
			json.dump(self.input, fp)


	def load_weights(self, model_name):
		file_name = "./ml_agent/saved_models/" + model_name + ".weights"
		#Load weigths of nn
		# self.agent.load_weights(file_name)


	@classmethod
	def from_file(cls, model_name):
		file_info = "./ml_agent/saved_models/"+ model_name + ".json"
		with open(file_info, 'r') as fr:
			dct = json.load(fr)
		rl_instance = cls(**dct)
		rl_instance.load_weights(model_name)
		return rl_instance

