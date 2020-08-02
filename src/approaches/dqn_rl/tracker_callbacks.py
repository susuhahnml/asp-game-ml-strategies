import pandas as pd
import timeit	
import numpy as np
import warnings
from csv import DictWriter
from rl.callbacks import Callback
from py_utils.logger import log

import os

class SaveTrackEpisodes(Callback):
    def __init__(self, name, net, save_every=None):
        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        # We therefore use a dictionary that is indexed by the episode to separate episodes
        # from each other.
        self.episode_start = {}
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.metrics = {}
        self.step = 0
        self.name = name
        self.net = net
        self.save_every = save_every

    def on_train_begin(self, logs):
        """ Print training values at beginning of training """
        self.train_start = timeit.default_timer()
        self.metrics_names = self.model.metrics_names
        self.all_info = []
        log.info('Training for {} steps ...'.format(self.params['nb_steps']))

    def on_episode_begin(self, episode, logs):
        """ Reset environment variables at beginning of each episode """
        self.episode_start[episode] = timeit.default_timer()
        self.episode_var_dict = {}
        self.episode_var_dict['episode'] = episode


        self.observations[episode] = []
        self.actions[episode] = []
        self.metrics[episode] = []
        self.rewards[episode] = []


    def on_step_end(self, step, logs):
        """ Update statistics of episode after each step """
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.actions[episode].append(logs['action'])
        self.metrics[episode].append(logs['metrics'])
        self.rewards[episode].append(logs['reward'])
        self.step += 1

    def on_episode_end(self, episode, logs):
        """ Compute and print training statistics of the episode when done """

    
        self.episode_var_dict['duration'] = timeit.default_timer() - self.episode_start[episode]
        self.episode_var_dict['steps'] = len(self.observations[episode])
        self.episode_var_dict['observation_hist'] = self.observations[episode]
        self.episode_var_dict['action_hist'] = self.actions[episode]
        self.episode_var_dict['final_action'] = self.actions[episode][-1]
        self.episode_var_dict['final_state'] = self.observations[episode][-1]
        self.episode_var_dict['reward'] = self.rewards[episode][-1]

        # Get metrics for loss, mae, and mean_q
        metrics = np.array(self.metrics[episode])
        metrics_variables = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for idx, name in enumerate(self.metrics_names):
                try:
                    value = np.nanmean(metrics[:, idx])
                except Warning:
                    value = '--'
                metrics_variables.append((name, value))      

        for tpl in metrics_variables:
            self.episode_var_dict[tpl[0]] = tpl[1]
        
        self.all_info.append(self.episode_var_dict)

            # Free up resources.
        del self.episode_var_dict
        del self.episode_start[episode]
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]

        if (not self.save_every is None):
            ep_nb = episode+1 if episode>0 else 0
            if (ep_nb%self.save_every==0):
                self.net.model=self.model.model 
                self.net.save_model(model_name='{}/{}'.format(self.net.model_name,int(ep_nb/self.save_every)))

    def on_train_end(self, logs):
        """ Print training time at end of training """
        duration = timeit.default_timer() - self.train_start
        self.save_to_file()
        log.info('Training finished, took {:.3f} seconds'.format(duration))

    def save_to_file(self):
        COLUMN_NAMES = ['episode', 'duration', 'steps', 'reward', 'loss', 'mae', 'mean_q', 'final_state', 'final_action', 'observation_hist', 'action_hist']
        file_name = "approaches/dqn_rl/saved_models/" + self.name + ".csv"

        try:
            with open(file_name, 'w') as csvfile:
                writer = DictWriter(csvfile, fieldnames = COLUMN_NAMES)
                writer.writeheader()
                for data in self.all_info:
                    writer.writerow(data)
        except IOError:
            log.error("Error saving")

        window_plot = 100
        df = pd.read_csv(file_name)
        df = df.groupby(np.arange(len(df))//window_plot).mean()
        df = df.reset_index()
        df['episodes']=df['index']*window_plot
        file_name_plot = "approaches/dqn_rl/saved_models/" + self.name + ".pdf"
        plot = df.plot(x='episodes', y='reward')
        os.makedirs(os.path.dirname(file_name_plot), exist_ok=True)
        plot.get_figure().savefig(file_name_plot, format='pdf')