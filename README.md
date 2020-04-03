## **Domiknows** 🧠:chart_with_upwards_trend:
## *Dominos machine learning strategies powered by ASP*

### Overview

In this project we train agents using supervised learning and reinforcement learning to play the game of Dominoes for two players. To do so, we use a framework with Answer Set Programming as engine to describe the game and its dynamics.

#### Game description

The description of the game is represented in ASP and it is called from python using Clingo API to compute the legal actions and successor states providing the game environment.
 
To represent the game encoding the framework requires the use of [Game Description Language (GDL)](https://en.wikipedia.org/wiki/Game_Description_Language), allowing the formalization of any finite game with complete information. 

#### Dominoes

Its a nice game

Some of the already available games are [Nim](https://en.wikipedia.org/wiki/Nim) and TicTacToe. 


### Learning approaches

#### Reinforcement learning

#### Supervised Learning

### Citations

Gebser, Kaminski, Kaufmann and Schaub, 2019 (Clingo)

```
@article{gebser2019multi,
  title={Multi-shot ASP solving with clingo},
  author={Gebser, Martin and Kaminski, Roland and Kaufmann, Benjamin and Schaub, Torsten},
  journal={Theory and Practice of Logic Programming},
  volume={19},
  number={1},
  pages={27--82},
  year={2019},
  publisher={Cambridge University Press}
}
```
Law, Russo and Broda, 2015 (ILASP)

```
@misc{ILASP_system,
  author="Law, Mark and Russo, Alessandra and Broda, Krysia",
  title="The {ILASP} system for learning Answer Set Programs",
  year="2015",
  howpublished={\url{www.ilasp.com}}
}
```

### Authors

Susana Hahn, Lukas

Cognitive Systems, University of Potsdam, WiSe 2019/20
