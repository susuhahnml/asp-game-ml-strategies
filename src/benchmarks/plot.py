import json
import matplotlib.pyplot as plt
from py_utils.logger import log
import os
import re
import pandas as pd

def plot_vs_benchmarks(files,args):
    all_bm = []
    for f in files:
        path ="benchmarks/vs/{}/{}".format(args.game_name,f)
        if os.path.isfile(path):
            with open(path) as feedsjson:
                j = json.load(feedsjson)
                all_bm.append(j)        
        else:
            for filename in os.listdir(path):
                with open(os.path.join(path, filename), 'r') as feedsjson: 
                    j = json.load(feedsjson)
                    all_bm.append(j)
    player_name = "a"
    numbers_games = set([j['args']['num_repetitions']*2 for j in all_bm])

    assert len(numbers_games)==1,"All benchmarks should play the same amount of games. Delete all benchmarks and images and run the vs script again."

    number_games = numbers_games.pop()
    data = make_dataframe(all_bm, number_games)

    if args.plot_type == "bar":
        plt = plot_barchart(data)
        plt.title(make_title(args))
        save_plot(plt, args)
    elif args.plot_type == "line":
        plt = plot_linegraph(data)
        plt.title(make_title(args))
        save_plot(plt, args)
    elif args.plot_type == "all":
        plt1 = plot_barchart(data)
        plt1.title(make_title(args))
        save_plot(plt1, args, plot_type="bar")
        plt2 = plot_linegraph(data)
        plt2.title(make_title(args))
        save_plot(plt2, args, plot_type="line")

    

def make_dataframe(dct, n_games):
    players = [model['results']["a"]['style_name'] for model in dct]
    wins = [int(model['results']["a"]['wins']) for model in dct]
    win_perc = [n_win/n_games for n_win in wins]
    illegal = [int(model['results']["a"]['matches_lost_by_illegal']) for model in dct]
    ill_perc = [n_ill/n_games for n_ill in illegal]
    d = {'player_name': players, 'wins': wins, "win_perc": win_perc, "illegal": illegal, "ill_perc":ill_perc, "n_games": n_games}
    df = pd.DataFrame(data=d)
    try:
        df['iter'] = [str(int(re.sub('[^\d*]', '', name))+1) for name in df["player_name"]]
        df = df.sort_values(by=['iter'])
        df['iter_int'] = [int(re.sub('[^\d*]', '', name))+1 for name in df["player_name"]]
    except:
        df['iter'] = df["player_name"]
        df['iter_int']  = df["player_name"]
    return df

def plot_barchart(df):
    if df.shape[0] > 20:
        plt.figure(figsize=(10, 5))
    try:
        df = df.sort_values(by=['iter_int'], ascending=True)
    except:
        pass
    width_c = df.shape[0]*0.03
    n_games = df["n_games"][0]
    pilegal = plt.bar(df['iter'], df['illegal'],bottom=df['wins'], width=width_c,color='#FC755A')
    pwin = plt.bar(df['iter'], df['wins'], width=width_c,color='#1644AE')
    plt.axhline(y = n_games/2, color ="black", linestyle ="--")

    if type(['iter_int']) == 'int':
        plt.xlabel("iteration")
    else:
        plt.xlabel("approach")
    plt.ylabel("number of games")
    plt.legend((pwin[0], pilegal[0]), ('Wins', 'Lost by Illegal'))
    plt.ylim((0, n_games))
    try:
        if df.shape[0] > 20:
            plt.xticks(df['iter'], [a.replace('_','\n').replace('-','\n') for a in df['iter_int']], fontsize=6)
        else:
            plt.xticks(df['iter'], [a.replace('_','\n').replace('-','\n') for a in df['iter_int']])
    except:
        pass 
    plt.grid(zorder=0,axis='y')

    return plt

def plot_linegraph(df):
    if df.shape[0] > 20:
        plt.figure(figsize=(15, 5))
    try:
        df = df.sort_values(by=['iter_int'], ascending=True)
    except:
        pass
    pwin = plt.plot(df['iter_int'], df['win_perc'],color='#1644AE')
    pilegal = plt.plot(df['iter_int'], df['ill_perc'],color='#FC755A')
    try:
        if df.shape[0] > 20:
            plt.xticks(df['iter_int'],[a.replace('_','\n').replace('-','\n') for a in df['iter_int']], fontsize=6)
        else:
            plt.xticks(df['iter_int'],[a.replace('_','\n').replace('-','\n') for a in df['iter_int']])
    except:
        pass 
    plt.axhline(y = 0.5, color ="black", linestyle ="--")   
    if type(['iter_int']) == 'int':
        plt.xlabel("iteration")
    else:
        plt.xlabel("approach")
    plt.ylabel("percentage")
    plt.legend((pwin[0], pilegal[0]), ('% games won', '% games lost with illegal action'))
    plt.ylim((-0.05, 1))
    return plt

def save_plot(plt, args, plot_type=None):
    if not plot_type:
        plot_type = args.plot_type
        file_out = "benchmarks/img/{}/{}_{}.png".format(args.game_name, args.plot_out, args.plot_type)
    else:
        file_out = "benchmarks/img/{}/{}_{}.png".format(args.game_name, args.plot_out, plot_type)
    os.makedirs(os.path.dirname(file_out), exist_ok=True)
    plt.savefig(file_out,dpi=200,bbox_inches='tight')
    log.info(plot_type + " plot saved in "+file_out)


def make_title(args):
    game_name = args.game_name.title()
    if game_name =="Dom":
        game_name = "Dominoes"
    approach_name = args.plot_out.title()
    return  " ".join([game_name, "results for", approach_name])