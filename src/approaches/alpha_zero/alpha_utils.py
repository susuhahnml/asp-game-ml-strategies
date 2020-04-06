import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model
from py_utils.logger import log
from structures.step import Step
def get_architecture(game_def,lr, architecture_name="dense"):
    action_size = game_def.encoder.action_size
    state_size = game_def.encoder.state_size
    inputs = Input(shape=(state_size,))
    hidden1 = Dense(120, activation='relu')(inputs)
    hidden2 = Dense(30, activation='relu')(hidden1)
    hidden3 = Dense(120, activation='relu')(hidden2)
    pi = Dense(action_size, activation='softmax', name='pi')(hidden3) 
    v = Dense(1, activation='tanh', name='v')(hidden3)         
    model = Model(inputs=inputs, outputs=[pi,v])
    model.compile(loss=['mean_squared_error','mean_squared_error'], optimizer=Adam(lr))
    model.summary()
    return model

def copy_model(game_def, model, lr):
    model_copy= clone_model(model)
    # model_copy.build((None, game_def.encoder.state_size)) 
    model_copy.compile(loss=['mean_squared_error','mean_squared_error'], optimizer=Adam(lr))
    model_copy.set_weights(model.get_weights())
    return model_copy

def train(model, examples, train_epochs, train_batch_size):   
    log.info("Training for {} epochs with batch size {}".format(train_epochs,train_batch_size))
    input_states, target_pis, target_vs = list(zip(*examples))
    input_states = np.asarray(input_states)
    target_pis = np.asarray(target_pis)
    target_vs = np.asarray(target_vs)
    # print(input_states)
    # print(target_pis)
    # print(target_vs)
    # print("Predicted-------")
    # print(model.predict(input_states))
    model.fit(x = input_states, y = [target_pis, target_vs], batch_size = train_batch_size, epochs = train_epochs)
    
def predict(game_def,state,model):
    state_masked = game_def.encoder.mask_state(state)
    pi, v = model.predict(np.array([state_masked]))
    return pi[0], v[0][0]