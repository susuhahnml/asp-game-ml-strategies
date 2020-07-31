import json
import matplotlib.pyplot as plt
from py_utils.logger import log
def plot_vs_benchmarks(files,args):
    all_bm = []
    for f in files:
        with open("benchmarks/vs/{}/{}".format(args.game_name,f)) as feedsjson:
            j = json.load(feedsjson)
            all_bm.append(j)

    player_name = "a"
    numbers_games = set([j['args']['num_repetitions']*2 for j in all_bm])
    assert len(numbers_games)==1,"All benchmarks should play the same amount of games"
    number_games = numbers_games.pop()
    players = [j['results'][player_name]['style_name'] for j in all_bm]
    wins = [int(j['results'][player_name]['wins']) for j in all_bm]
    illegal = [int(j['results'][player_name]['matches_lost_by_illegal']) for j in all_bm]
    
    pwin = plt.bar(players, wins, width=0.3,color='#1644AE')
    pilegal = plt.bar(players, illegal,bottom=wins, width=0.3,color='#FC755A')
    plt.axhline(y = number_games/2, color ="black", linestyle ="--")
    plt.xticks(rotation='vertical')
    plt.xlabel("Players")
    plt.ylabel("Games VS Random")
    plt.legend((pwin[0], pilegal[0]), ('Wins', 'Lost by Illegal'))
    plt.ylim((0, number_games))
    plt.grid(zorder=0,axis='y')
    log.info("Plot saved in benchmarks/plot.png")
    plt.savefig("benchmarks/plot.png",dpi=200,bbox_inches='tight')
