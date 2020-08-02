import json
import matplotlib.pyplot as plt
from py_utils.logger import log
import os
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
    assert len(numbers_games)==1,"All benchmarks should play the same amount of games"
    number_games = numbers_games.pop()
    players = [j['results'][player_name]['style_name'] for j in all_bm]
    wins = [int(j['results'][player_name]['wins']) for j in all_bm]
    illegal = [int(j['results'][player_name]['matches_lost_by_illegal']) for j in all_bm]
    
    width_c = len(players)*0.03

    pwin = plt.bar(players, wins, width=width_c,color='#1644AE')
    pilegal = plt.bar(players, illegal,bottom=wins, width=width_c,color='#FC755A')
    plt.axhline(y = number_games/2, color ="black", linestyle ="--")
    plt.xticks(rotation='vertical')
    

    plt.xlabel("Players")
    plt.ylabel("Games VS Random")
    plt.legend((pwin[0], pilegal[0]), ('Wins', 'Lost by Illegal'))
    plt.ylim((0, number_games))
    plt.grid(zorder=0,axis='y')
    file_out = "benchmarks/img/{}/{}.png".format(args.game_name, args.plot_out)
    os.makedirs(os.path.dirname(file_out), exist_ok=True)
    plt.savefig(file_out,dpi=200,bbox_inches='tight')
    log.info("Plot saved in "+file_out)
