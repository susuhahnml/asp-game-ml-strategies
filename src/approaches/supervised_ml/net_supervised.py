        
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
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import pdb
from tensorflow.keras.models import model_from_json
import os
from structures.net import Net

class NetSupervised(Net):

    approach = "supervised_ml"
    def __init__(self, game_def, model_name, model=None, args={}):
        super().__init__(game_def, model_name, model,args)


    def load_model_from_args(self):    
        action_size = self.game_def.encoder.action_size
        state_size = self.game_def.encoder.state_size
        if self.args.architecture_name=="default":
            model = Sequential()
            model.add(Dense(40,input_dim=action_size))
            model.add(Activation('relu'))
            model.add(Dense(state_size))
            model.add(Activation('softmax'))
            return model
        else :
            raise NotImplementedError("Architecture named {} is not defined ".format(self.args.architecture_name))

    def compile_model(self,model):
        if model is None:
            raise RuntimeError("A loaded model is required for compiling")
        model.compile(loss=['mean_squared_error','mean_squared_error'], optimizer=Adam(self.args.lr))

    def train(self):
        if self.model is None:
            raise RuntimeError("A loaded model is required for training")
        # action_size = game_def.encoder.action_size
        # input_size = action_size + game_def.encoder.state_size
        # training_path = './approaches/supervised_ml/train/{}/{}'.format(game_def.name, training_file)
        # train_data, test_data = divide_data(training_path,input_size,action_size)

        # # Train the base model to predict next state
        # base_model = get_architecture(input_size,action_size,architecture_name)

        # base_model.compile(loss='categorical_crossentropy',
        #             optimizer='adam',
        #             metrics=['acc']) 
        # log.info("Fittin model for {} epochs".format(epochs))
        # base_model.fit(train_data[0], train_data[1], epochs=epochs, batch_size=10, verbose=0)

        # #Test base model
        # scores = base_model.evaluate(test_data[0], test_data[1])
        # log.info("Base model scores -> %s: %.2f%%" % (base_model.metrics_names[1], scores[1]*100))

        # # Remove last layer
        # base_model.pop()
        # base_model.pop()

        # # Replace with v: Outcome of the game r: Reward of the game
        # out_v = Dense(1, activation='tanh', name='v')(base_model.layers[-1].output)
        # out_r = Dense(1, activation='tanh', name='r')(base_model.layers[-1].output)
        # final_model = Model(base_model.input,[out_r,out_v])

        # #Train final model
        # final_model.compile(loss=['mean_squared_error','mean_squared_error'],
        #     optimizer='adam',
        #     metrics=['acc'])
        # label = [train_data[2][:,0],train_data[2][:,1]]
        # final_model.fit(train_data[0], label, epochs=epochs, batch_size=10, verbose=0)

        # #Test final model
        # label_test = [test_data[2][:,0],test_data[2][:,1]]
        # scores = final_model.evaluate(test_data[0], label_test)
        # log.info("Final model scores -> %s: %.2f%%" % (final_model.metrics_names[1], scores[1]*100))
        # print(final_model.predict(train_data[0]))

    
    def predict_single(self, values):
        raise NotImplementedError