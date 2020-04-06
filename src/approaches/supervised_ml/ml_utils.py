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

def divide_data(training_path, input_size,action_size):
    pandas_train = pd.read_csv(training_path,sep=';')
    pandas_train = pandas_train.sample(frac=1).reset_index(drop=True)
    train, test = train_test_split(pandas_train,test_size = 0.1)
    # print(train.iloc[0][train.iloc[0]!=0])
    train = train.to_numpy()
    test = test.to_numpy()

    train_inputs = train[:,0:input_size]
    train_next_label = train[:,input_size:input_size+action_size]
    train_pred_label = train[:,-2:]

    test_inputs = test[:,0:input_size]
    test_next_label = test[:,input_size:input_size+action_size]
    test_pred_label = test[:,-2:]
    return (train_inputs,train_next_label,train_pred_label), (test_inputs,test_next_label,test_pred_label)


def train(game_def, architecture_name, epochs, training_file):
    action_size = game_def.encoder.action_size
    input_size = action_size + game_def.encoder.state_size
    training_path = './approaches/supervised_ml/train/{}/{}'.format(game_def.name, training_file)
    train_data, test_data = divide_data(training_path,input_size,action_size)

    # Train the base model to predict next state
    base_model = get_architecture(input_size,action_size,architecture_name)

    base_model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['acc']) 
    log.info("Fittin model for {} epochs".format(epochs))
    base_model.fit(train_data[0], train_data[1], epochs=epochs, batch_size=10, verbose=0)

    #Test base model
    scores = base_model.evaluate(test_data[0], test_data[1])
    log.info("Base model scores -> %s: %.2f%%" % (base_model.metrics_names[1], scores[1]*100))

    # Remove last layer
    base_model.pop()
    base_model.pop()

    # Replace with v: Outcome of the game r: Reward of the game
    out_v = Dense(1, activation='tanh', name='v')(base_model.layers[-1].output)
    out_r = Dense(1, activation='tanh', name='r')(base_model.layers[-1].output)
    final_model = Model(base_model.input,[out_r,out_v])

    #Train final model
    final_model.compile(loss=['mean_squared_error','mean_squared_error'],
        optimizer='adam',
        metrics=['acc'])
    label = [train_data[2][:,0],train_data[2][:,1]]
    final_model.fit(train_data[0], label, epochs=epochs, batch_size=10, verbose=0)

    #Test final model
    label_test = [test_data[2][:,0],test_data[2][:,1]]
    scores = final_model.evaluate(test_data[0], label_test)
    log.info("Final model scores -> %s: %.2f%%" % (final_model.metrics_names[1], scores[1]*100))
    print(final_model.predict(train_data[0]))


    return final_model

def get_architecture(input_size,output_size,architecture_name="dense"):
    #TODO add different arch with an if
    if architecture_name=="dense":
        model = Sequential()
        model.add(Dense(40,input_dim=input_size))
        model.add(Activation('relu'))
        model.add(Dense(output_size))
        model.add(Activation('softmax'))
        return model