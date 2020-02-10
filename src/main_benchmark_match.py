#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import time
import argparse
import time
import json
import random
from py_utils import arg_metav_formatter, add_params
from game_definitions import GameDef
from py_utils.match_simulation import simulate_match
from tqdm import tqdm
from py_utils.logger import log
from structures.players import Player

def benchmark_match(n,name_a,name_b,random_seed,game_name,constants):
    log.info("Benchmarking: {} vs {} for {} games".format(name_a,name_b,n))
    results = ""
    game = GameDef.from_name(game_name,constants)
    player_encounters = [[
        Player.from_config({'name':name_a,'main_player':'a'}, game_def=game),
        Player.from_config({'name':name_b,'main_player':'b'}, game_def=game)
    ],[
        Player.from_config({'name':name_b,'main_player':'a'}, game_def=game),
        Player.from_config({'name':name_a,'main_player':'b'}, game_def=game)
    ]]
    game.get_random_initial()
    initial_states = game.random_init
    random.Random(random_seed).shuffle(initial_states)
    scores = [{'wins':0,'losses':0,'draws':0,'points':0,'response_time':0},{'wins':0,'losses':0,'draws':0,'points':0,'response_time':0}]
    for turn, vs in enumerate(player_encounters):
        idx = {'a':0+turn,'b':1-turn}
        for i in tqdm(range(n)):
            game.initial = initial_states[i]
            match, metrics = simulate_match(game,vs,ran_init=False)
            goals = match.goals
            for l,g in goals.items():
                scores[idx[l]]['points']+=g
                if g>0:
                    scores[idx[l]]['wins']+=1
                elif g<0:
                    scores[idx[l]]['losses']+=1
                else:
                    scores[idx[l]]['draws']+=1
            scores[idx['a']]['response_time']+=metrics['a']  
            scores[idx['b']]['response_time']+=metrics['b']  
        scores[idx['a']]['response_time']=int(scores[idx['a']]['response_time']/n)
        scores[idx['b']]['response_time']=int(scores[idx['b']]['response_time']/n)

    results += ("------------------\nGame:{}".format(game.__class__.__name__))

    results += """
    {}:
        wins: {}   losses: {}   points: {} response_time(ms): {}
    {}:
        wins: {}   losses: {}   points: {} response_time(ms): {}
    """.format(name_a,scores[0]['wins'],scores[0]['losses'],scores[0]['points'],scores[0]['response_time'],
    name_b,scores[1]['wins'],scores[1]['losses'],scores[1]['points'],scores[1]['response_time'])
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=arg_metav_formatter)
    params = ["--log",
            "--n-match",
            "--pA-style",
            "--pB-style",
            "--game-name",
            "--const",
            "--file-name",
            "--random-seed",
            "--grid-search"
            ]
    add_params(parser,params)
    args = parser.parse_args()
    log.set_level(args.log)
    if args.const is None:
        constants = {}
    else:
        constants = {c.split("=")[0]:c.split("=")[1] for c in args.const}


    if(args.grid_search):
        args.file_name = "grid-search.txt"
        players = ['rl-dense-0.1', 'rl-dense-0.3', 'strategy', 'random'] 
        res = ""
        for i in range(len(players)):
            for j in range(i+1,len(players)):
                result = benchmark_match(args.n_match,players[i],players[j],args.random_seed,args.game_name,constants)
                res += result
    else:
        res = benchmark_match(args.n_match,args.pA_style,args.pB_style,args.random_seed,args.game_name,constants)
    
    log.info(res)
    args.file_name = "benchmarks/"+args.file_name
    with open(args.file_name, "w") as text_file:
        text_file.write(res)
        log.info("Results saved in " + args.file_name)

    

   
