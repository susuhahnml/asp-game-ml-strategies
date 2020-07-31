        
from tensorflow.keras.layers import Input
import os.path
import csv
from py_utils.logger import log
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score, KFold
from sklearn.utils.multiclass import type_of_target
from py_utils.logger import log
import numpy as np
import asyncio
import json
import tensorflow as tf
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import pdb
from tensorflow.keras.models import model_from_json
import os
from structures.net import Net
from approaches.dqn_rl.gym_env import GymEnv
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent
from rl.core import Processor
import numpy as np
tf.compat.v1.disable_eager_execution()
from approaches.dqn_rl.tracker_callbacks import SaveTrackEpisodes

import matplotlib.pyplot as plt

class CustomProcessor(Processor):
    '''
    acts as a coupling mechanism between the agent and the environment
    '''

    def process_state_batch(self, batch):
        '''
        Given a state batch, I want to remove the second dimension, because it's
        useless and prevents me from feeding the tensor into my CNN
        '''
        return np.squeeze(batch, axis=1)
        
class NetDQN(Net):

    approach = "dqn_rl"
    def __init__(self, game_def, model_name, model=None, args={}, possible_initial_states=[]):
        super().__init__(game_def, model_name, model,args)
        self.possible_initial_states=possible_initial_states


    def load_model_from_args(self):    
        action_size = self.game_def.encoder.action_size
        state_size = self.game_def.encoder.state_size
        arch = self.args.architecture_name
        if arch=="default":
            inputs = Input(shape=(state_size,))
            hidden1 = Dense(120, activation='relu', activity_regularizer=l2(0.0001))(inputs)
            hidden2 = Dense(30, activation='relu', activity_regularizer=l2(0.0001))(hidden1)
            hidden3 = Dense(120, activation='relu', activity_regularizer=l2(0.0001))(hidden2)
            hidden4 = BatchNormalization()(hidden3)
            pi = Dense(action_size, activation='softmax', name='pi')(hidden4)
            model = Model(inputs=inputs, outputs=pi)
            self.model = model
            # self.compile_model(self.model)
            #Is compiled in agent
        else :
            raise NotImplementedError("Architecture named {} is not defined ".format(self.args.architecture_name))

    def compile_model(self,model):
        if model is None:
            raise RuntimeError("A loaded model is required for compiling")
        model.compile(loss='mean_squared_error', optimizer=Adam(self.args.lr))


    def train(self):
        if self.model is None:
            raise RuntimeError("A loaded model is required for training")

        #Params: eps, number od steps
        action_size = self.game_def.encoder.action_size
        input_size = action_size + self.game_def.encoder.state_size

        env = GymEnv(self.game_def,self.possible_initial_states,opponent=self.args.strategy_opponent)
        policy = EpsGreedyQPolicy(eps = self.args.eps)  
        memory = SequentialMemory(limit=50000, window_length=1)      
        agent = DQNAgent(model=self.model, nb_actions=action_size, memory=memory, nb_steps_warmup=100, target_model_update=1e-2, policy=policy, processor=CustomProcessor())
        agent.compile(Adam(lr=1e-3), metrics=['mae'])
		
        training_logger = SaveTrackEpisodes(name="{}/{}".format(self.game_def.name,self.model_name),net=self)		   
        agent.fit(env, nb_steps=self.args.nb_steps, visualize=False, nb_max_episode_steps=99,verbose=1, callbacks=[training_logger])
        self.model = agent.model
        
    
    def predict_single(self, value):
        return self.model.predict(np.array([value]))


    def predict_state(self, state):
        """
        Makes a prediction for a single state
        """
        if self.model is None:
            raise RuntimeError("A loaded model is required for predicting")
        state_masked = self.game_def.encoder.mask_state(state)
        return self.predict_single(state_masked)


    def predict_pi_v(self,state):
        pi = self.predict_state(state)
        return pi[0],1