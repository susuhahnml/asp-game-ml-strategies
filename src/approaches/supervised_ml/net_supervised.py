        
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
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import pdb
from tensorflow.keras.models import model_from_json
import os
from structures.net import Net
from approaches.supervised_ml.utils import *

class NetSupervised(Net):

    approach = "supervised_ml"
    def __init__(self, game_def, model_name, model=None, args={}):
        super().__init__(game_def, model_name, model,args)


    def load_model_from_args(self):    
        action_size = self.game_def.encoder.action_size
        state_size = self.game_def.encoder.state_size
        if self.args.architecture_name=="default":
            #Loads only first model
            dyn_model = Sequential()
            dyn_model.add(Dense(120,input_dim=action_size+state_size, activation="relu", activity_regularizer=l2(0.01)))
            dyn_model.add(Dropout(0.25))
            dyn_model.add(Dense(state_size, activation='sigmoid'))
            self.model = dyn_model
            self.compile_model(self.model)
        else : 
            raise NotImplementedError("Architecture named {} is not defined ".format(self.args.architecture_name))

    def compile_model(self,model):
        if model is None:
            raise RuntimeError("A loaded model is required for compiling")
        model.compile(loss='binary_crossentropy',
                    optimizer='adam', # Root Mean Square Propagation
                    metrics=['acc']) # Accuracy performance metric

    def train(self):
        if self.model is None:
            raise RuntimeError("A loaded model is required for training")
        action_size = self.game_def.encoder.action_size
        state_size = self.game_def.encoder.state_size
        dyn_model = self.model
        file_name = self.args.training_file
        training_file_path = "approaches/supervised_ml/train_data/{}/{}".format(self.game_def.name,file_name)
        train_data, test_data = divide_train_data(training_file_path,action_size, state_size, clean=True)

        es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=50, min_delta=0.0005)
        
        log.info("Training dynamics model...")
        dyn_history = dyn_model.fit(train_data["input"], train_data["next"], epochs=50, batch_size=50, verbose=0, validation_split=0.1, callbacks=[es])

        file_name_plot = "approaches/supervised_ml/saved_models/{}/{}_dynamic.pdf".format(self.game_def.name,self.model_name)
        fig = show_acc_loss_curves(dyn_history)
        fig.savefig(file_name_plot, format='pdf')

        results = test_model(dyn_model, test_data, test="next")
        log.info("Dynamics Network----- loss: {}, acc: {}".format(results[0],results[1]))

        # set hyperparameter search space
        # optimiser = ["nag","adam"]
        # learning_rate = [0.0001, 0.001, 0.01, 0.1]
        # batch_size = [10, 500, 100, 500]
        # reg_penalty = [0.01, 0.001, 0.0001]
        # max_epochs = [5000]
        # transfer = [True, False]
        # add_layers = [True, False]
        optimiser = ["adam"]
        learning_rate = [0.001]
        batch_size = [500]
        reg_penalty = [0.001]
        max_epochs = [5000]
        transfer = [True]
        add_layers = [True, False]
        # create list of all different parameter combinations
        param_grid = dict(optimiser = optimiser, learning_rate = learning_rate, batch_size = batch_size, 
                        reg_penalty = reg_penalty, epochs = max_epochs, transfer = transfer, add_layers = add_layers)
        combinations = list(product(*param_grid.values()))


        model,history = run_3_fold_gridsearch(train_data, test_data, combinations, "grid_search_reg.csv", dyn_model)

        file_name_plot = "approaches/supervised_ml/saved_models/{}/{}.pdf".format(self.game_def.name, self.model_name)
        fig = show_acc_loss_curves(history)
        fig.savefig(file_name_plot, format='pdf')



        loss_test, acc_test = test_model(model, test_data, test="pred")    
        loss_train, acc_train = test_model(model, train_data, test="pred")    

        log.info("Final on train: loss: {} acc: {}".format(loss_train,acc_train))
        log.info("Final on test: loss: {} acc: {}".format(loss_test,acc_test))
        self.model = model
        

    
    def predict_single(self, value):
        return self.model.predict(np.array([value]))


    def predict_state(self, state):
        """
        Makes a prediction for a single state
        """
        if self.model is None:
            raise RuntimeError("A loaded model is required for predicting")
        state_masked = self.game_def.encoder.mask_state(state)
        outputs = []
        for a in self.game_def.encoder.all_actions:
            input_net = np.concatenate((state_masked,self.game_def.encoder.mask_action(str(a))))
            outputs.append(self.predict_single(input_net)[0][0])
        return outputs

    def predict_pi_v(self,state):
        pi = self.predict_state(state)
        return pi,1