import os.path
import csv
from py_utils.logger import log
import numpy as np
import asyncio
import json
import pandas as pd
import pdb
import os
from tensorflow.keras.models import model_from_json


def save_model(model, model_name, game_name, approach):
    file_base = "./approaches/{}/saved_models/{}".format(approach,game_name)

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


def load_model_from_name(model_name, game_name,approach):
    file_base = "./approaches/{}/saved_models/{}/{}".format(approach,game_name, model_name)

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


def training_data_to_csv(file_name, training_list, game_def, new_files,extra_array=['reward','win']):
    obs = [str(o) for o in game_def.encoder.all_obs["a"]]
    COLUMN_NAMES = ["'INIT:{}'".format(o) for o in obs]
    COLUMN_NAMES.extend([str(o) for o in game_def.encoder.all_actions])
    COLUMN_NAMES.extend(["'NEXT:{}'".format(o) for o in obs])
    COLUMN_NAMES.extend(extra_array)
    
    try:

        exists = os.path.isfile(file_name)

        with open(file_name, new_files) as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            if new_files=="w":
                writer.writerow(COLUMN_NAMES)
            for l in training_list:
                row = []
                control = l['s_init'].control
                current_state = l['s_init']
                row.extend(game_def.encoder.mask_state(current_state,control))
                row.extend(game_def.encoder.mask_action(str(l['action'].action)))
                current_state = l['s_next']
                row.extend(game_def.encoder.mask_state(current_state,control))
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
