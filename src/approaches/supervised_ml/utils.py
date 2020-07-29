import warnings  
import traceback
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import keras.backend as K
    from tensorflow.keras.callbacks import EarlyStopping
    import tensorflow as tf
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.models import Sequential, Model, clone_model
    from tensorflow.keras.regularizers import l2

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from itertools import product
from py_utils.logger import log

rnd_state = 1

"""
reads in training data from file and divides it into input state, next state and probability of winning
outputs a dictionary with training and testing data
"""

def divide_train_data(training_path, action_size, state_size, clean = False, test_size = 0.1):
    pandas_train = pd.read_csv(training_path,sep=';')
    if clean:
        pandas_train = clean_data(pandas_train,action_size, state_size)
    pandas_train = pandas_train.sample(frac=1, random_state=rnd_state).reset_index(drop=True)
    train, test = train_test_split(pandas_train,test_size = test_size, shuffle=False)
    log.info("With a test size of {}, we get {} training and {} testing samples.".format(test_size, train.shape[0], test.shape[0]))
    a_s_size = action_size+state_size
    train_inputs = train.iloc[:,0:a_s_size]
    train_next_label = train.iloc[:,a_s_size:a_s_size+state_size]
    train_pred_label = train.iloc[:,-2:-1]
    test_inputs = test.iloc[:,0:a_s_size]
    test_next_label = test.iloc[:,a_s_size:a_s_size+state_size]
    test_pred_label = test.iloc[:,-2:-1]
    train_dict = {"input": train_inputs.to_numpy(), "next":train_next_label.to_numpy(), "pred": train_pred_label.to_numpy()}
    test_dict = {"input": test_inputs.to_numpy(), "next": test_next_label.to_numpy(), "pred": test_pred_label.to_numpy()}
    return train_dict, test_dict

"""
removes duplicate instances of the same state-input-state sequences keeping only the most-visited states
"""
def clean_data(dataframe,action_size, state_size):
    log.info("Cleaning data...")
    cleaned_df = dataframe
    cleaned_df_str = convert_cols(cleaned_df, str)
    cleaned_df_str['combo'] = cleaned_df_str.apply(lambda x: ''.join(x), axis=1)
    cleaned_df_str['combo'] = cleaned_df_str['combo'].str[:action_size+state_size]
    cleaned_df_str = cleaned_df_str.sort_values('n', ascending=False).drop_duplicates(['combo'])
    cleaned_df_str = cleaned_df_str.drop('combo', 1)
    cleaned_df = convert_cols(cleaned_df_str, float)
    diff = dataframe.shape[0] - cleaned_df.shape[0]
    log.info("Removed {} duplicates with different probability scores".format(diff))
    log.info("This leaves {} instances for training and testing".format(cleaned_df.shape[0]))
    return cleaned_df

"""
converts columns (optinally specified by indices) in a dataframe into the specified data type
"""
def convert_cols(dataframe, datatype, indices=None):
    res_df = dataframe
    if indices:
        for col in list(res_df.columns.values)[indices]:
            res_df[col] = res_df[col].astype(datatype)
    else:
        for col in list(res_df.columns.values):
            res_df[col] = res_df[col].astype(datatype)
    return res_df


"""
draws accuracy and loss curves for training
"""
def show_acc_loss_curves(hist, metric = "acc"):
    if metric == "mea":
        draw_dict = {"loss": {"full": "loss"}, "mea_metric": {"full": "mea_metric"}}
    elif metric == "acc":
        draw_dict = {"loss": {"full": "loss"}, "acc": {"full": "accuracy"}}
    elif metric == "both":
        {"loss": {"full": "loss"}, "mea_metric": {"full": "mea_metric"}, "acc": {"full": "accuracy"}}
        
    fig, axes = plt.subplots(nrows=1, ncols=len(draw_dict.keys()), figsize=(15, 5))

    
    for i, key in enumerate(draw_dict.keys()):
        axes[i].plot(hist.history[key])
        axes[i].plot(hist.history['val_'+key])
        axes[i].set_title('model '+draw_dict[key]["full"])
        axes[i].set_ylabel(key)
        axes[i].set_xlabel('epoch')
        axes[i].legend(['train', 'test'], loc='upper left')
    fig.tight_layout()

    return fig

"""
evaluates model for either next state (test="next") or probablity of winning (test="pred") and return evaluation metrics
"""
def test_model(model, test_data, test):
    return model.evaluate(test_data["input"], test_data[test], verbose=0)


# define cross entropy
def cross_entropy(y_true, y_pred):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

"""
creates deep copy of a model including its weights
"""
def copy_model(model, optimiser, set_weights,learning_rate=0.001):
    model_copy= clone_model(model)
    opt = Adam(learning_rate=learning_rate)
        
    model_copy.compile(loss=cross_entropy, optimizer=opt, metrics=["acc"])
    if set_weights:
        model_copy.set_weights(model.get_weights())
    return model_copy

"""
creates model based on given input parameters
"""
def create_model(optimiser, learning_rate, c, transfer_learning, deepen_model, base_model):
    # transfer learning condition
    model = copy_model(base_model, optimiser, transfer_learning)
    model.pop()
    if deepen_model:
        if transfer_learning:
            model.layers[0].trainable = False
            start = Dense(120, activation='relu', activity_regularizer=l2(c))(model.layers[-1].output)
            end = Dense(250, activation='relu', activity_regularizer=l2(c))(start)
            out_r = Dense(1, activation="sigmoid", activity_regularizer=l2(c), name='r_prop')(end)
            final_model = Model(model.input,out_r)
        else:
            final_model = Sequential()
            final_model.add(Dense(120,input_dim=51, activation="relu", activity_regularizer=l2(c)))
            final_model.add(Dense(250, activation='relu', activity_regularizer=l2(c)))
            final_model.add(Dense(1, activation="sigmoid", activity_regularizer=l2(c), name='r_prop'))
    else:
        out_r = Dense(1, activation="sigmoid", activity_regularizer=l2(c),
                      name='r_prop')(model.layers[-1].output)
        final_model = Model(model.input,out_r)
        
    # Compile model
    if optimiser == "nag":
        opt = SGD(learning_rate=learning_rate, nesterov=True)
    elif optimiser == "adam":
        opt = Adam(learning_rate=learning_rate)
    final_model.compile(loss=cross_entropy, optimizer=opt, metrics=["acc"])
    
    return final_model

def run_3_fold_gridsearch(train_data, test_data, combinations, filename, dyn_model):
    # create containers for resulting data
    res_df = pd.DataFrame(columns=['transfer','deepened','optimiser','learning rate','batch size',
                                   'loss1', 'acc1','loss2', 'acc2','loss3', 'acc3'])
    hist_dict_global = {}

    num_splits = 3
    
    best_model = None
    best_acc = 0
    best_history = []
    best_params = []
    # 3-fold grid search over the combinations defined above
    for i, combination in enumerate(combinations):

        kf = KFold(n_splits=num_splits, random_state=42, shuffle=False)
        metrics_dict = {}

        log.info("{}/{}: {} - folds completed: ".format(i+1,len(combinations), combination))

        acc_total = 0
        for j, (train_index, test_index) in enumerate(kf.split(train_data["input"])):
            log.info("starting folding {}".format(j))
            X_train, X_test = train_data["input"][train_index], train_data["input"][test_index]
            y_train, y_test = train_data["pred"][train_index], train_data["pred"][test_index]

            model = create_model(optimiser = combination[0], learning_rate = combination[1], 
                                    c = combination[3], transfer_learning = combination[5], 
                                    deepen_model = combination[6], base_model = dyn_model)
            es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=80, min_delta=0.001)
            hist = model.fit(X_train, y_train, epochs=combination[4], batch_size=combination[2],
                                verbose=0, use_multiprocessing = True, callbacks=[es],validation_split=0.1)

            # try to evaluate the model
            loss, acc = model.evaluate(X_test, y_test, verbose=0)
            acc_total +=acc
            metrics_dict[j+1] = {"loss": loss, "acc": acc, "epoch_stopped": es.stopped_epoch}            
        
        acc = acc_total/num_splits
        if acc>best_acc:
            log.info("New best model with acc {}".format(acc))
            best_model = model
            best_acc = acc
            best_history = hist
            best_params=combination

        row = {'transfer': combination[5], 'deepened': combination[6], 'optimiser': combination[0], 
               'learning rate': combination[1], 'batch size': combination[2], 'reg_penalty': combination[3],
               'epoch_stopped1': metrics_dict[1]["epoch_stopped"], 'loss1': metrics_dict[1]["loss"], 
               'acc1': metrics_dict[1]["acc"],
               'epoch_stopped2': metrics_dict[2]["epoch_stopped"], 'loss2': metrics_dict[2]["loss"], 
               'acc2': metrics_dict[2]["acc"],
               'epoch_stopped3': metrics_dict[3]["epoch_stopped"], 'loss3': metrics_dict[3]["loss"], 
               'acc3': metrics_dict[3]["acc"]}
        res_df = res_df.append(row , ignore_index=True)
        res_df.to_csv(filename, sep=";")

    log.info("Best model found using parameters:")
    print(best_params)
    return best_model,best_history

def plot_train_grid_search():
    col_list = ['optimiser', 'transfer', 'deepened', 'learning rate', 'batch size', 'loss_mean', 'loss_std',
        'loss_min', 'acc_mean', 'acc_std', 'epochs_mean', 'epochs_std']
    #Plot results of grid search
    data_raw = pd.read_csv("grid_search_reg.csv", sep=";", index_col=0)
    data_raw.shape
    data = data_raw.copy()

    data["loss_mean"] = data[['loss1', 'loss2', 'loss3']].mean(axis=1, skipna = True)
    data["loss_std"] = data[['loss1', 'loss2', 'loss3']].std(axis=1, skipna = True)
    data["acc_mean"] = data[['acc1', 'acc2', 'acc3']].mean(axis=1, skipna = True)
    data["acc_std"] = data[['acc1', 'acc2', 'acc3']].std(axis=1, skipna = True)
    data["epochs_mean"] = data[['epoch_stopped1', 'epoch_stopped2', 'epoch_stopped3']].mean(axis=1, skipna = True)
    data["epochs_std"] = data[['epoch_stopped1', 'epoch_stopped2', 'epoch_stopped3']].std(axis=1, skipna = True)
    # get minimum value
    data["loss_min"] = data[["loss1","loss2", "loss3"]].min(1)


    data.sort_values(axis=0, by="loss_mean")[col_list].head(20)

    #Show plots

    g = sns.FacetGrid(data, col = "batch size", sharey=True, aspect=1.5, margin_titles=True)
    g.map(sns.barplot, "learning rate", "loss_mean", order = [0.0001, 0.001, 0.01, 0.1])
    plt.show()

    g = sns.FacetGrid(data, col="transfer", sharey=True, aspect=1.5, margin_titles=True)
    g.map(sns.barplot, "deepened", "loss_mean")
    plt.show()

    g = sns.FacetGrid(data, col="transfer", sharey=True, aspect=1.5, margin_titles=True)
    g.map(sns.barplot, "deepened", "epochs_mean")
    plt.show()