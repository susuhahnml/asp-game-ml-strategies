import os.path
import csv
from approaches.rl_agent.game import Game
from py_utils.logger import log
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score, KFold
from sklearn.utils.multiclass import type_of_target

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
import pdb
from tensorflow.keras.models import model_from_json
import os

def save_model(model, model_name, game_name):
    file_base = "./approaches/ml_agent/saved_models/{}".format(game_name)

    file_weights = "{}/{}.h5".format(file_base,model_name)
    file_model = "{}/{}.json".format(file_base,model_name)
    os.makedirs(os.path.dirname(file_weights), exist_ok=True)
    os.makedirs(os.path.dirname(file_model), exist_ok=True)

    # serialize model to JSON
    model_json = model.to_json()
    with open(file_model, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(file_weights)
    log.info("Model saved in {}".format(file_base))


def load_model_from_name(model_name, game_name):
    file_base = "./approaches/ml_agent/saved_models/{}/{}".format(game_name, model_name)

    file_model = file_base + ".json"
    file_weights = file_base + ".h5"

    # load json and create model
    json_file = open(file_model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(file_weights)
    log.info("Model loaded from {}".format(file_base))

    return loaded_model
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


def train(game_def, architecture, epochs, training_file):
    game = Game(game_def)
    action_size = game.nb_actions
    input_size = action_size + game.nb_observations
    training_path = './approaches/ml_agent/train/{}/{}'.format(game_def.name, training_file)
    train_data, test_data = divide_data(training_path,input_size,action_size)

    # Train the base model to predict next state
    base_model = ModelSelector(input_size,action_size).return_model(architecture)

    base_model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['acc']) 
    log.info("Fittin model for {} epochs".format(epochs))
    base_model.fit(train_data[0], train_data[1], epochs=epochs, batch_size=10, verbose=0)

    #Test base model
    scores = base_model.evaluate(test_data[0], test_data[1])
    log.info("Base model scores -> %s: %.2f%%" % (base_model.metrics_names[1], scores[1]*100))
    print(base_model.predict(train_data[0]))

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


    # kfold = KFold(n_splits=3)
    # cvscores = []

    # for train, test in kfold.split(train_data[0], train_data[1]):
    # 	log.info("Doing fold for {} epochs".format(epochs))
    # 	# Fit the model
    # 	base_model.fit(train_data[0].iloc[train,:], train_data[1].iloc[train,:], epochs=epochs, batch_size=10, verbose=0)
    # 	# evaluate the model
    # 	scores = base_model.evaluate(train_data[0].iloc[test,:], train_data[1].iloc[test,:], verbose=0)
    # 	log.info("%s: %.2f%%" % (base_model.metrics_names[1], scores[1]*100))

    # 	cvscores.append(scores[1] * 100)
    # log.info("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


def training_data_to_csv(file_name, training_list, game_def, new_files):
    games = {'a': Game(game_def,main_player="a"),
    "b": Game(game_def,main_player="b")}

    obs = [str(o) for o in games['a'].all_obs]
    act = [str(o) for o in games['a'].all_actions]
    COLUMN_NAMES = ["'INIT:{}'".format(o) for o in obs]
    COLUMN_NAMES.extend(act)
    COLUMN_NAMES.extend(["'NEXT:{}'".format(o) for o in obs])
    COLUMN_NAMES.extend(["reward","win"])

    try:

        exists = os.path.isfile(file_name)

        with open(file_name, new_files) as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            if new_files=="w":
                writer.writerow(COLUMN_NAMES)
            for l in training_list:
                row = []
                control = l['s_init'].control
                games[control].current_state = l['s_init']
                row.extend(games[control].current_observation)
                row.extend(games[control].mask_action(str(l['action'].action)))
                games[control].current_state = l['s_next']
                row.extend(games[control].current_observation)
                for k in extra_array:
                    row.extend([l[k]])
                writer.writerow([int(r) for r in row[:-len(extra_array)]]+[r for r in row[-len(extra_array):]])
    except IOError as e:
        log.error("Error saving csv")
        log.error(e)

def remove_duplicates_training(file_name):
    csv_file = open(file_name, "r")
    lines = csv_file.read().split("\n")
    csv_file.close()
    writer = open(file_name, "w")
    lines_set = set(lines)
    log.info("Removing duplicates in {}, from {} to {} lines".format(file_name,len(lines),len(lines_set)))
    for line in lines_set:
        writer.write(line + "\n")
    writer.close()
