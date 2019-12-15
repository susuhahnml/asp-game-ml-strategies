#!/usr/bin/env python
# -*- coding: utf-8 -*-

import clingo
import operator
from structures.match import *
from py_utils.colors import *
from structures.tree import Tree
from anytree import RenderTree

case = {
    'a':{
        'a': {
            'optimization': "#maximize{N,T:holds(goal(a,N),T)}.",
            'current_is_better': operator.gt
        },
        'b': {
            'optimization': "#minimize{N,T:holds(goal(a,N),T)}.",
            'current_is_better': operator.lt
        }
    },
    'b':{
        'a': {
            'optimization': "#minimize{N,T:holds(goal(b,N),T)}.",
            'current_is_better': operator.lt
        },
        'b': {
            'optimization': "#maximize{N,T:holds(goal(b,N),T)}.",
            'current_is_better': operator.gt
        }
    },
}

def get_match(game_def, optimization, fixed_atoms, learned_rules, main_player):
    """
    Makes a clingo call to compute the match for the fixed atoms 
    and the player optimization.
    """
    ctl = clingo.Control("0")
    # Check if it can load from grounded atoms gotten from AS
    ctl.load(game_def.full_time)
    ctl.add("base",[],fixed_atoms)
    ctl.add("base",[],"".join(learned_rules))
    ctl.add("base",[],optimization)
    ctl.ground([("base", [])], context=Context())
    with ctl.solve(yield_=True) as handle:
        matches = []
        models = []
        for model in handle:
            matches.append(Match.from_time_model(model,game_def,main_player))
            models.append(model)
        if len(matches) == 0:
            return None
        return matches[-1]

def get_minmax_init(game_def, main_player, initial, learning_rules = True, debug = False):
    """
    Computes the minmax using multiple calls to clingo.
    Args:
        - game_def: Game definition 
        - main_player: The name of the main player, must have control in the initial state
        - initial: The initial state
    Returns:
        - Tuple (minmax_match, minmax_tree, examples_list)
        - minmax_match: A match representing the minmax match correspunding to the best scenario when the other player tries to win.
        - minmax_tree: The search tree with the parts that where requeried to compute
        - examples_list: The list of examples for ILASP
    """
    match  = get_match(game_def,case[main_player][main_player]['optimization'],initial,[],'a')
    examples_list = []
    learned_rules = []
    node = Tree.node_from_match_initial(match)
    minmax_match, minmax_tree = get_minmax_rec(game_def,match,node,0,main_player,learned_rules= learned_rules,list_examples= examples_list,learning_rules = learning_rules, debug=debug)
    minmax_tree=Tree(minmax_tree.parent)
    final_score = minmax_match.goals[main_player]
    minmax_match.steps[0].set_score(final_score)
    minmax_tree.root.name.set_score(final_score)
    return minmax_match, minmax_tree, examples_list

def get_minmax_rec(game_def, match, node_top, top_step, main_player, old_fixed='', learned_rules=[], level = 0, list_examples=[], learning_rules=True, debug = False):
    """
    Computes the recursive call to compute the minmax with asp
    Args:
        - game_def: Game definition
        - match: The calculated match that we wish to prune into a real minmax
        - node_top: The node that corresponds to the top of the tree
        - top_step: Number indicating the maximum level to reach going up, fro which the actions are fixed
        - main_player: The player we aim to maximize.
        - old_fixed: The fixed facts to represent the position on the tree search,
        - learned_rules: The rules learned for the tree
        - level: The recursion level
    """
    steps_to_analyze = match.steps[top_step:-1]
    minmax_match = match
    node_top = node_top.leaves[0]
    for step in steps_to_analyze[::-1]:
        node_top = node_top.parent
        i = step.time_step
        control = step.state.control
        ##Fix all current and explore other actions
        fixed = old_fixed
        fixed += ''.join(s.to_asp_syntax() for s in match.steps[0:i])
        fixed += match.steps[i].fluents_to_asp_syntax()
        fixed += 'not ' + match.steps[i].action_to_asp_syntax()
        current_goal = minmax_match.goals[main_player]
        #Get optimal match (Without minimizing for second player)
        opt_match  = get_match(game_def,fixed,case[main_player][control]['optimization'],learned_rules,main_player)
        # Score is current goal unless proved other
        node_top.name.set_score(current_goal)
        if(not opt_match):
            # No more actions possible
            continue
        #Compute tree for optimal match
        opt_node = Tree.node_from_match(Match(opt_match.steps[i:]))
        opt_node.parent = node_top.parent
        new_goal = opt_match.goals[main_player]
        if case[main_player][control]['current_is_better'](current_goal,new_goal):
            # Choosing other action gets a worst result in the best case
            ex = generate_example(match.steps[i].state,match.steps[i].action,opt_match.steps[i].action)
            list_examples.append(ex)
            if learning_rules:
                rule = generate_rule(game_def,match.steps[i].state,match.steps[i].action)
                learned_rules.append(rule)
            # Minmax was fixed, set score without minimizing
            opt_node.name.set_score(new_goal)
            continue
        if new_goal == current_goal:
            # Other action is as best as good as this one, this section will remain unexplored
            continue
        ##### New match is potentially better
        #Get real minmax for optimal match (minimize for other player)
        opt_minmax, opt_minmax_tree = get_minmax_rec(game_def,opt_match, opt_node, i,main_player,fixed,learned_rules,level+1,list_examples,debug=debug)
        opt_minmax_tree.parent = node_top.parent
        new_goal = opt_minmax.goals[main_player]
        if case[main_player][control]['current_is_better'](current_goal,new_goal):
            # Choosing other action gets a worst result in the best case
            ex = generate_example(match.steps[i].state,match.steps[i].action,opt_minmax.steps[i].action)
            list_examples.append(ex)
            if learning_rules:
                rule = generate_rule(game_def,match.steps[i].state,match.steps[i].action)
                learned_rules.append(rule)
            continue
        if new_goal == current_goal:
            # Other action is as best as good as this one, this section will remain unexplored
            continue
        #### New match is better for current player maximized
        ex = generate_example(match.steps[i].state,opt_minmax.steps[i].action,match.steps[i].action)
        list_examples.append(ex)
        if learning_rules:
            rule = generate_rule(game_def,match.steps[i].state,opt_minmax.steps[i].action)
            learned_rules.append(rule)
        node_top.parent.name.set_score(new_goal)
        #Update minmax match
        minmax_match = opt_minmax
    return minmax_match, node_top
