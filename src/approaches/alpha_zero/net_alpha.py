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
import pdb
from tensorflow.keras.models import model_from_json
import os
from structures.net import Net
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2

def cross_entropy(y_true, y_pred):
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)


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
            hidden1 = Dense(120, activation='relu', activity_regularizer=l2(0.0001))(inputs)
            hidden2 = Dense(30, activation='relu', activity_regularizer=l2(0.0001))(hidden1)
            hidden3 = Dense(120, activation='relu', activity_regularizer=l2(0.0001))(hidden2)
            hidden4 = BatchNormalization()(hidden3)
            pi = Dense(action_size, activation='linear', name='pi_non_softmaxed')(hidden4) 
            v = Dense(1, activation='tanh', name='v')(hidden4)         
            # pi = Dense(action_size, activation='softmax', name='pi')(hidden3)
            # v = Dense(1, activation='tanh', name='v')(hidden4)
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
        pi_softmaxed = tf.nn.softmax(pi[0])
        pi_softmaxed = pi_softmaxed.numpy()
        return pi_softmaxed, v[0][0]