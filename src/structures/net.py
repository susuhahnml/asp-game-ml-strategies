        
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
import pdb
from tensorflow.keras.models import model_from_json
import os
from tensorflow.keras.models import clone_model
# tf.compat.v1.disable_eager_execution()
from keras.backend import manual_variable_initialization 
manual_variable_initialization(True)
class Net():
    """
    A class used to represent a network, this class can be extended by special 
    networks by implementing the needed methods.

    Attributes
    ----------
    approach: The name of the approach using the network
    """
    approach = "none"

    def __init__(self, game_def, model_name, model=None,args={}):
        """
        Constructs a network for the given game.

        Args:
            game_def (GameDef): The game definition used for the creation
            model_name (str): The name of the network
            model (Tensorflow): A model for the network. It can also be
                                loaded latter from a file, using load_model_from_file
            args (Dic): Dictionary with the required arguments to create
                        a model, these arguments can come from the command line, using load_model from args,
        """
        self.game_def = game_def
        self.file_base = "./approaches/{}/saved_models/{}".format(self.__class__.approach,game_def.name)
        self.game_name = game_def.name
        self.model_name = model_name
        self.model = model
        self.args = args

    def load_model_from_file(self):   
        """
        Loads the model from a file using the model_name attribute.
        """
        path = '{}/{}'.format(self.file_base,self.model_name)

        file_weights = "{}.h5".format(path)
        file_model = "{}.json".format(path)

        # load json and create model
        json_file = open(file_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(file_weights)
        log.info("Model loaded from {}".format(path))
        self.model = loaded_model

    def load_model_from_args(self):
        """
        Loads the model using the attribute args. 
        This method should construct the model and compile it
        """
        raise NotImplementedError
    
    def compile_model(self,model):
        """
        Function to compile a given model

        Args:
        model (Tensorflow) The model to be compiled
        """
        raise NotImplementedError
        
    def copy_model(self):
        """
        Returns a copy of the networks model compiled.
        """
        model_copy= clone_model(self.model)
        self.compile_model(model_copy)
        model_copy.set_weights(self.model.get_weights())
        return model_copy
    
    def copy(self):
        """
        Returns a copy of the complete network
        """
        model_copy= self.copy_model()
        new_copy =  self.__class__(self.game_def,self.model_name,model_copy,self.args)
        return new_copy


    def save_model(self, model_name=None):
        """
        Saves the model using its model name
        Args:
            model_name (str): Value to overwrite the name for saving
        """
        model_name = self.model_name if model_name is None else model_name
        path = '{}/{}'.format(self.file_base,model_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
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
        log.info("Model saved in {}".format(path))

    def train(self):
        """
        Trains the network. It must ensure that 
        the model was previously loaded.
        """
        raise NotImplementedError
    
    def predict_single(self, value):
        """
        Makes a prediction for a single value using the model
        """
        raise NotImplementedError
