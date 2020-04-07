        
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
from structures.net import Net
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

def cross_entropy(y_true, y_pred):
    eps = 1e-15
    # clipping is used to ensure that we never take the log of exactly 1 or 0
    p_pred = K.clip(y_pred, eps, 1 - eps)
    p_true = K.clip(y_true, eps, 1 - eps) 
    return -K.mean((p_true*K.log(p_pred))+((1-p_true)*K.log(1-p_pred)))

class NetAlpha(Net):

    approach = "alpha_zero"
    def __init__(self, game_def, model_name, model=None, args={}):
        super().__init__(game_def, model_name, model, args)


    def load_model_from_args(self):    
        action_size = self.game_def.encoder.action_size
        state_size = self.game_def.encoder.state_size
        arch = self.args.architecture_name
        if arch=="default":
            inputs = Input(shape=(state_size,))
            hidden1 = Dense(120, activation='relu')(inputs)
            hidden2 = Dense(30, activation='relu')(hidden1)
            hidden3 = Dense(120, activation='relu')(hidden2)
            pi = Dense(action_size, activation='softmax', name='pi')(hidden3) 
            v = Dense(1, activation='tanh', name='v')(hidden3)         
            model = Model(inputs=inputs, outputs=[pi,v])
            self.model = model
            self.compile_model(self.model)
            self.model.summary
        else :
            raise NotImplementedError("Architecture named {} is not defined ".format(arch))

    def compile_model(self,model):
        if model is None:
            raise RuntimeError("A loaded model is required for compiling")
        if self.args.loss == 'custom':
            losses ={'pi':cross_entropy,
            'v':'mean_squared_error'
            }
        else:
            losses ={'pi':self.args.loss,
            'v':'mean_squared_error'
            }

        lossWeights={'pi':0.5,
          'v':0.5  
        }
        model.compile(optimizer=Adam(self.args.lr),
             loss=losses,
             loss_weights= lossWeights
        )

        
    def train(self, examples = []):
        if self.model is None:
            raise RuntimeError("A loaded model is required for training")
        log.info("Training for {} epochs with batch size {}".format(self.args.n_epochs,self.args.batch_size))
        
        input_states, target_pis, target_vs = list(zip(*examples))
        input_states = np.asarray(input_states)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        history = self.model.fit(x = input_states, y = [target_pis, target_vs], batch_size = self.args.batch_size, epochs = self.args.n_epochs,verbose=0)
        log.info("Initial loss: {}  Final loss: {}".format(history.history["loss"][0],history.history["loss"][-1]))
    
    def predict_state(self, state):
        if self.model is None:
            raise RuntimeError("A loaded model is required for predicting")
        state_masked = self.game_def.encoder.mask_state(state)
        pi, v = self.model.predict(np.array([state_masked]))
        return pi[0], v[0][0]