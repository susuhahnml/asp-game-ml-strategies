        
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
tf.compat.v1.disable_eager_execution()
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential, Model
import pdb
from tensorflow.keras.models import model_from_json
import os
from tensorflow.keras.models import clone_model


class Net():

    approach = "none"

    def __init__(self, game_def, model_name, model=None,args={}):
        self.game_def = game_def
        self.file_base = "./approaches/{}/saved_models/{}".format(self.__class__.approach,game_def.name)
        self.game_name = game_def.name
        self.model_name = model_name
        self.model = model
        self.args = args

    def load_model_from_file(self):    
        file_weights = "{}/{}.h5".format(self.file_base,self.model_name)
        file_model = "{}/{}.json".format(self.file_base,self.model_name)

        # load json and create model
        json_file = open(file_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(file_weights)
        log.info("Model loaded from {}".format(self.file_base))
        self.model = loaded_model

    def load_model_from_args(self):    
        raise NotImplementedError
    
    def compile_model(self,model):
        raise NotImplementedError
        
    def copy_model(self):
        model_copy= clone_model(self.model)
        self.compile_model(model_copy)
        model_copy.set_weights(self.model.get_weights())
        return model_copy
    
    def copy(self):
        print("old")
        print(self.model)
        model_copy= self.copy_model()
        new_copy =  self.__class__(self.game_def,self.model_name,model_copy,self.args)
        print("new")
        print(new_copy.model)
        return new_copy


    def save_model(self, model_name=None):
        model_name = self.model_name if model_name is None else model_name
        file_weights = "{}/{}.h5".format(self.file_base,model_name)
        file_model = "{}/{}.json".format(self.file_base,model_name)
        os.makedirs(os.path.dirname(file_weights), exist_ok=True)
        os.makedirs(os.path.dirname(file_model), exist_ok=True)

        # serialize model to JSON
        model_json = self.model.to_json()
        with open(file_model, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(file_weights)
        log.info("Model saved in {}".format(self.file_base))

    def train(self):
        #must train the model
        raise NotImplementedError
    
    def predict_single(self, values):
        raise NotImplementedError
