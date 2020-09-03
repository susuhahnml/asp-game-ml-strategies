## Game strategies powered by ASP  :clubs: :game_die:

Here we document our source code and its various functionalities. 

### Table of contents

1. [Dependencies](#1-Dependencies) 
2. [Code structure](#2-Code-structure)
3. [Integration of learning approaches](#3-Integration)
4. [Main](#4-Main)
   a. [Build an approach](#Build-approach)
   b. [Simulate match](#Simulate-match)

### 1. Dependencies

#### i. Pythonic dependencies 

To install python-based dependencies, one can use the file `requirements.txt`. Simply execute the following (possibly within a virtual environment):

```shell
$ pip install -r requirements.txt
```

Additionally, for conducting sanity tests, we recommend installing `pytest`:

```shell
$ pip install pytest
```

#### ii. External dependencies

The ASP system [clingo](https://potassco.org/clingo/) is required for the game dynamics. This system can be installed using `conda` with the command:

```shell
$ conda install -c potassco clingo
```



### 2. Code structure

Here we provide a tabular summary of our main code structure.

#### i. `game_definitions`

| L1                   | L2                                          | L3                                                  | Description                                                                                                                                 |
| :---:                | :---:                                       | :---:                                               | :---                                                                                                                                        |
| **game_definitions** |                                             |                                                     | Contains files with the asp encodings of games                                                                                              |
|                  | **game_name**                               |                                                     | Generic name of game (eg. "nim"). Used in the command line                                                                 |
|                  |                                         | [background.lp](game_definitions/nim/background.lp) | Defines the basic rules using of the game for one `time_step`, using `next` With GDL syntax                                                                |
|                  |                                         | [default_initial.lp](game_definitions/nim/default_initial.lp)       | Defines the initial state of the game used as default                                                                                                       |
|                  |                                         | [all_init.lp](game_definitions/nim/all_init.lp)       | Defines all possible initial states. One per stable model                                                                                                       |


#### ii. `approaches`

| L1                   | L2                                          | L3                                                  | Description                                                                                                                                 |
| :---:                | :---:                                       | :---:                                               | :---                                                                                                                                        |
| **approaches** |                                             |                                                     | Contains the a folder for each strategic approach to play a game                                                                                              |
|                  | **approach_name**                               |                                                     | Generic name of approach (eg. "min_max"). Used in the command line                                                                 |
|                  |                                         | [player.lp](approaches/min_max/player.py) | Defines how a player using this approach should be built and how it makes a desition given a state of the game                      |


#### iii. `benchmarks`

| L1           | L2                                                        | Description                                |
| :---:        | :---:                                                     | :---                                       |
| **benchmarks** |                                                           | Folders with output benchmarks files               |
|          | **approach_name**                               |                                                     | Will store all benchmarks generated for the building of 'approach _name'                                                                |
|                  |                                         | **game_name**       | The name of the game for the benchmarks                                                                                                       |
|          | **vs**                               |                                                     | Will store all benchmarks generated when players play against each other                                                               |
|                  |                                         | **game_name**       | The name of the game for the benchmarks                                                                                                       |


#### iv. `py_utils`

| L1           | L2                                                        | Description                                |
| :---:        | :---:                                                     | :---                                       |
| **py_utils** |                                                           | Folders with utils functions               |
|          | [arg_metav_formatter.py](py_utils/arg_metav_formatter.py) | Argparse formatter for cli information     |
|          | [clingo_utils.py](py_utils/clingo_utils.py)               | Clingo bindings to be used in python with CLingo API       |
|          | [colors.py](py_utils/colors.py)                           | Defining python colors for pretty-printing |
|          | [logger.py](py_utils/logger.py)                           | Defines the logs of the framework |
|          | [train_utils.py](py_utils/train_utils.py)                           | Common functions for training |

#### v. `structures`

| L1             | L2                                | Description                                                                                                                                                                          |
| :---:          | :---:                             | :---                                                                                                                                                                                 |
| **structures** |                                   | Contains the structures used to model the games                                                                                                                                      |
|            | [action.py](structures/action.py) | An action selected by a player. An extended class also includes the fluents of the next state once the action ins performed                                                          |
|            | [game_def.py](structures/game_def.py)  | Defines the general class for game definitions                                     |
|            | [game_encoder.py](structures/game_encoder.py)  | Defines the class used to vectorize a game state                                    |
|            | [match.py](structures/match.py)   | A full match of a game, list of steps                                                                                                                                                |
|            | [net.py](structures/net.py)   | A general class to manage networks                                                                                                                                               |
|            | [players.py](structures/players.py)  | Defines the general behavior of  of a player approach                                    |
|            | [state.py](structures/state.py)   | The state of the game, including board state, hows turn it is, if the game finished and if such, the goals reached. An extended class also includes all valid actions from the state |
|            | [step.py](structures/step.py)     | The step on a match, includes the state and the action performed in such state                                                                                                       |
|            | [tree.py](structures/tree.py)     | A game tree created by steps, with all possible paths                                                                                                                      |
|            | [treeMCTS.py](structures/tree_MCTS.py)     | A Monte carlo tree search tree.                                                                                                                    |
|            | [tree_net.py](structures/tree_net.py)     | A game tree generated by a network.                                                                                                                    |

### 3. Integration of learning approaches

We consider as a learning approach, a process that is capable of creating a strategy given a game description and use such strategy latter on to choose actions during game play. A strategy might involve an additional ASP file, a pre-computed tree search, a machine learning model among many other.

All approaches can be found inside the [approaches](src/approaches) directory. Every folder in this directory will automatically generate command-line arguments to run the building of the strategy and to play such approach against other, generating usefully benchmarks.

The instructions to create a new strategic approach can be found [here](src/approaches/README.md). Please refer to the README.md file in each approach for any specific information.

### 4. Main

We provide three types of main functions. Both of them have the following arguments in common to define the game, the number of times they will be ran and where the benckmark's will be saved. All command must me used under the `src` directory.

```shell
$ python main.py -h
```

```
--log LOG             Log level: 'info' 'debug' 'error'
--game-name GAME_NAME
                      Short name for the game. Must be the name of the
                      folder with the game description
--const CONST         A constant for the game definition that will passed to
                      clingo on every call. Must of the form <id>=<value>,
                      can appear multiple times
--random-initial-state-seed RANDOM_INITIAL_STATE_SEED
                      The initial state for each repetition will be
                      generated randomly using this seed. One will be
                      generated for each repetition. This requires the game
                      definition to have a file named rand_initial.lp as
                      part of its definition to generate random states.
--initial-state-full-path INITIAL_STATE_FULL_PATH
                      The full path starting from src to the file considered
                      for the initial state. Must have .lp extension
--num-repetitions NUM_REPETITIONS
                      Number of times the process will be repeated
--benchmark-output-file BENCHMARK_OUTPUT_FILE
                      Output file to save the benchmarks of the process that
                      was ran
```

#### Build your approach

This command can be used to build an approach. It will run the `bild()` method of the player's approach. Using its own specific arguments.
Usage:
```shell
$ python main.py <approach-name>
```
Example:
```shell
$ python main.py min_max -h
```

The help will include the arguments specific for this approach:  
```
--tree-image-file-name TREE_IMAGE_FILE_NAME
                        Name of the file save an image of the computed tree
--main-player MAIN_PLAYER
                        The player for which to maximize; either a or b
```


It will give benchmarks for build times and special results. Benchmarks are saved in JSON format in specific folders inside [benchamrks](/src/benchmarks) and appended into the [vs json](/src/benchmarks/build.json).



#### Simulate a match

This command can be used to simulate a match with between two previously loaded players in an specific game definition. It requires the following additional arguments to define the kind of players.

Example:
```shell
$ python main.py vs -h
```

```
--pA-style PA_STYLE   Playing style name for player a;
--pB-style PB_STYLE   Playing style name for player b;
--play-symmetry       When this flag is passed, all games will be played
                      twice, one with player a starting and one with player
                      b starting to increase fairness
```

This command will generate games playing the approaches against each other. It will give benchmarks regarding, winning, points and response times. Benchmarks are saved in JSON format in specific folders inside [benchamrks](/src/benchmarks) and appended into the [vs json](/src/benchmarks/vs.json).

Example:
```shell
$ python main.py vs --pA-style='strategy' --pB-style='random' --num-repetition=5 --game-name="nim"
```

```
INFO:  Using default initial state ./game_definitions/nim/default_initial.lp

INFO:  Benchmarking: strategy vs random for 5 games
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 47.27it/s]

INFO:
		Initial state:
		• • •
		• •
		•


		                strategy:
		                    wins: 2
		                    points: -1
		                    response_times(ms): [0.009, 0.004, 0.003, 0.002, 0.002]
		                random:
		                    wins: 3
		                    points: 1
		                    response_times(ms): [0.022, 0.014, 0.015, 0.01, 0.009]
```


#### Load a pre-trained player

A saved player can be also loaded for further visualization or any specific analysis. For this, we use the command:

```shell
pytho main.py load --style=<Name of saved model>
```


**Note** We ask the user to refer to the help on the command line to up to date features.