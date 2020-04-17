#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import numpy as np
import sys
from tqdm import tqdm
import time
import inspect
import random
import os
import re
from py_utils import CustomHelpFormatter
from structures.game_def import GameDef
import argparse
from py_utils.logger import log
from structures.players import player_approaches_sub_classes, Player
from structures.match import Match
import signal

def add_default_params(parser):
    parser.add_argument("--log", type=str, default="INFO",
        help="Log level: 'info' 'debug' 'error'" )
    parser.add_argument("--game-name", type=str, default="nim",
        help="Short name for the game. Must be the name of the folder with the game description")
    parser.add_argument("--const", type=str, action='append',
        help="A constant for the game definition that will passed to clingo on every call. Must of the form <id>=<value>, can appear multiple times")
    initial_group = parser.add_mutually_exclusive_group()
    initial_group.add_argument("--random-initial-state-seed", "--rand", type=int, default=None,
        help="The initial state for each repetition will be generated randomly using this seed. One will be generated for each repetition. This requires the game definition to have a file named rand_initial.lp as part of its definition to generate random states.")
    initial_group.add_argument("--initial","--init", type=str, default=None,
        help="The name of the file with the initial state inside the game definition.")
    parser.add_argument("--num-repetitions","--n", type=int, default=1,
        help="Number of times the process will be repeated")
    parser.add_argument("--benchmark-output-file", "--out",type=str, default="console",
        help="Output file to save the benchmarks of the process that was runned")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    # ---------------------------- Default parser ----------------------------
    add_default_params(parser)
    subs = parser.add_subparsers(help="Approach to build. Use 'vs' to play previously built players against each other",dest="selected_approach")
    player_classes = player_approaches_sub_classes()
    
    # ---------------------------- VS parser ----------------------------
    parser_vs = subs.add_parser('vs', 
        help="Plays one player approach against other and generates benchmarks",conflict_handler='resolve',formatter_class=CustomHelpFormatter)
    add_default_params(parser_vs)
    
    # ----- Get help for player styles
    player_name_style_options = []
    for n,pc in player_classes.items():
        player_name_style_options.append("{}:\t{}".format(n.upper(),getattr(pc,"get_name_style_description")()))

    parser_vs.add_argument("--pA-style","--a", type=str, default="random",
        help="R|Playing style name for player a:\n• "+ "\n•  ".join(player_name_style_options))
    parser_vs.add_argument("--pB-style","--b", type=str, default="random",
        help="R|Playing style name for player b:\n• "+ "\n•  ".join(player_name_style_options))
    parser_vs.add_argument("--play-symmetry", default=False, action='store_true',
        help="When this flag is passed, all games will be played twice, one with player a starting and one with player b starting to increase fairness")
    parser_vs.add_argument("--time-out-sec", type=int, default=3, 
        help="Number of seconds for player timeout when choosing action")

    # ---------------------------- Load parser ----------------------------
    parser_load = subs.add_parser('load', 
        help="Loads one player",conflict_handler='resolve',formatter_class=CustomHelpFormatter)
    add_default_params(parser_load)
    
    parser_load.add_argument("--style", type=str, default="random",
        help="R|Playing style name for player :\n• "+ "\n•  ".join(player_name_style_options))

    # ---------------------------- Parser for each approach ----------------------------
    
    for n, pc in player_classes.items():
        approach_parser = subs.add_parser(n,help=pc.description,formatter_class=CustomHelpFormatter)
        add_default_params(approach_parser)
        getattr(pc,"add_parser_build_args")(approach_parser)
    

    # ---------------------------- Setting default arguments ----------------------------
    args = parser.parse_args()
    n = args.num_repetitions
    log.set_level(args.log.upper())
    if args.const is None:
        constants = {}
    else:
        constants = {c.split("=")[0]:c.split("=")[1] for c in args.const}

    game_def = GameDef.from_name(args.game_name,constants=constants)
    using_random = not args.random_initial_state_seed is None
    using_fixed_initial = not args.initial is None
    if(using_random):
        log.info("Using random seed {} for initial states".format(args.random_initial_state_seed))
        game_def.get_random_initial()
        initial_states = game_def.random_init
        random.Random(args.random_initial_state_seed).shuffle(initial_states)
    elif(using_fixed_initial):
        log.info("Using fixed initial state {}".format(args.initial))
        initial_states = [game_def.path + "/"+args.initial]
    else:
        log.info("Using default initial state {}".format(game_def.initial))
        initial_states = [game_def.initial]

    # ---------------------------- Computing VS ----------------------------

    if args.selected_approach == 'vs':
        style_a = args.pA_style
        style_b = args.pB_style
        log.info("Benchmarking: {} vs {} for {} games".format(style_a,style_b,n))

        player_encounters = [[
            Player.from_name_style(game_def,style_a,'a'),
            Player.from_name_style(game_def,style_b,'b')
        ]]
        if(args.play_symmetry):
            player_encounters.append([
            Player.from_name_style(game_def,style_b,'a'),
            Player.from_name_style(game_def,style_a,'b')
        ])
        
        benchmarks= Match.vs(game_def,n,player_encounters,initial_states,[style_a,style_b],time_out_sec=args.time_out_sec)
    
    # ---------------------------- Computing Load ----------------------------

    if args.selected_approach == 'load':
        style = args.style
        log.info("Loading player {}".format(style))
        player = Player.from_name_style(game_def,style,'a')
        player.show_info(initial_states)
        benchmarks ={}

    # ---------------------------- Computing Build for Approach ----------------------------

    else :
        p_cls = player_classes[args.selected_approach]
        build_times = []
        for i in tqdm(range(n)):
            game_def.initial = initial_states[i%len(initial_states)]
            t0 = time.time()
            results= p_cls.build(game_def,args)
            t1 = time.time()
            build_times.append(round((t1-t0)*1000,3))

        build_times_np = np.array(build_times)
        benchmarks= {
                    'player': args.selected_approach,
                    'build':build_times,
                    'average_build':round(np.mean(build_times),3),
                    'std':round(np.std(build_times),3),
                    'special_results':results}
        if "rules_file_name" in args:
            if not args.rules_file_name is None:
                benchmarks['player'] = benchmarks['player']+'_learning'
    # ---------------------------- Saving Benchamarks ----------------------------
    command = ' '.join(sys.argv[1:])
    benchmarks_final= {
        'command':command,
        'args':vars(args),
        'initial_state': ".".join(game_def.get_initial_state().fluents_str) if not using_random else 'RANDOM',
        'results': benchmarks
    }
    json_benchmarks = json.dumps(benchmarks_final, indent=4)

    benchmark_file = args.benchmark_output_file
    if(benchmark_file == 'console'):
        log.info(json_benchmarks)
    else:
        benchmark_file = "benchmarks/{}/{}/{}".format(args.selected_approach,game_def.name,benchmark_file)
        os.makedirs(os.path.dirname(benchmark_file), exist_ok=True)
        with open(benchmark_file, "w") as text_file:
            text_file.write(json_benchmarks)
            log.info("Results saved in " + benchmark_file)   
    json_file = "benchmarks/vs.json" if args.selected_approach == 'vs' else 'benchmarks/build.json'
    
    with open(json_file) as feedsjson:
        full_arr = json.load(feedsjson)
    full_arr.append(benchmarks_final)
    with open(json_file, mode='w') as feedsjson:
        json.dump(full_arr, feedsjson, indent=4)

