import random
from structures.players import Player
from py_utils.logger import log
from py_utils.colors import bcolors, paint
from structures.players import Player
from py_utils.logger import log
from py_utils.colors import bcolors, paint
from structures.players import Player, IllegalActionError
import numpy as np
from structures.match import Match
from approaches.alpha_zero.tree_zero import TreeZero
from structures.tree_net import TreeNet
from structures.step import Step
from approaches.alpha_zero.net_alpha import NetAlpha
from random import randint

class AlphaZero(Player):

    """
    A player using the AlphaZero algorithm

    Attributes
    ----------
    game_def: The game definition
    main_player: The name of the player to optimize
    """

    description = "A player trained using alpha zero"

    def __init__(self, game_def, name_style, main_player, net=None,visualizing_net=True):
        """
        Constructs a player using saved information, the information must be saved 
        when the build method is called. This information should be able to be accessed
        using parts of the name_style to refer to saved files or other conditions

        Args:
            game_def (GameDef): The game definition used for the creation
            main_player (str): The name of the player either a or b
            name_style (str): The name style used to create the built player. This name will be passed
                        from the command line. 
        """
        name = "{}".format(name_style)
        super().__init__(game_def, name, main_player)
        if net is None:
            model_name = name_style[11:]
            self.net = NetAlpha(game_def,model_name)
            self.net.load_model_from_file()
        else:
            self.net = net
        self.visualizing_net=visualizing_net



    @classmethod
    def get_name_style_description(cls):
        """
        Gets a description of the required format of the name_style.
        This description will be shown on the console.
        Returns: 
            String for the description
        """
        return "alpha_zero-<file-name> where file-name indicates the name of the saved model inside ml_agent/saved_models/game_name"

    @staticmethod
    def match_name_style(name_style):
        """
        Verifies if a name_style matches the approach

        Args:
            name_style (str): The name style used to create the built player. This name will be passed
                        from the command line. Will then be used in the constructor. 
        Returns: 
            Boolean value indicating if the name_style is a match
        """
        return name_style[:11]=="alpha_zero-"


    @staticmethod
    def add_parser_build_args(approach_parser):
        """
        Adds all required arguments to the approach parser. This arguments will then 
        be present on when the build method is called.
        
        Args:
            approach_parser (argparser): An argparser used from the command line for
                this specific approach. 
        """
        approach_parser.add_argument("--architecture-name", type=str, default="default",
                            help="underlying neural network architecture-name;" +
                            " Available: 'default'")
        approach_parser.add_argument("--loss", type=str, default="custom",
                            help="The loss used for the probabilities. By default using custom cross-entropy")
        approach_parser.add_argument("--n-train", type=int, default=200,
                            help="Number of times the network will be trained and tested")
        approach_parser.add_argument("--n-episodes", type=int, default=100,
                            help="Number episodes computed with MCTS to get training examples")
        approach_parser.add_argument("--n-epochs", type=int, default=1500,
                            help="Epochs for each training")
        approach_parser.add_argument("--batch-size", type=int, default=200,
                            help="Batch size for each training")
        approach_parser.add_argument("--lr", type=float, default=0.01,
                            help="Learning rate for training")
        approach_parser.add_argument("--n-vs", type=int, default=150,
                            help="Number of matches to compare networks")
        approach_parser.add_argument("--n-mcts-simulations", type=int, default=300,
            help="Number of times the MCTS algorithm will transverse to compute probabilities")
        approach_parser.add_argument("--model-name", type=str, default="unnamed",
                            help="Name of the model, used for saving and logging")
        approach_parser.add_argument("--train-rand", type=int, default=None,
        help="Random seed for creating initial states in training net")
        approach_parser.add_argument("--vis-tree", default=False, action='store_true',
        help="A visualization of each tree will be saved in every improved network")


    @staticmethod
    def build(game_def, args):
        """
        Runs the required computation to build a player. For instance, creating a tree or 
        training a model. 
        The computed information should be stored to be accessed latter on using the name_style
        Args:
            game_def (GameDef): The game definition used for the creation
            args (NameSpace): A name space with all the attributes defined in add_parser_build_args
        """
        best_net = NetAlpha(game_def,args.model_name,model=None,args=args)
        best_net.load_model_from_args()
        game_def.get_random_initial()
        using_random = not args.train_rand is None
        if(using_random):
            log.info("Using random seed {} for initial states in training".format(args.train_rand))
            game_def.get_random_initial()
            initial_states = game_def.random_init
            random.Random(args.train_rand).shuffle(initial_states)
        else:
            log.info("Using default initial state in training {} ".format(game_def.initial))
            initial_states = [game_def.initial]

        number_initial =  len(initial_states) if args.n_vs>len(initial_states) else args.n_vs
        for i in range(args.n_train):
            log.info("------- Iteration {} --------".format(i))
            training_examples = []
            for e in range(args.n_episodes):
                log.debug("\t\tEpisode {}...".format(e))
                new_examples = TreeZero.run_episode(game_def, best_net)
                training_examples+=new_examples
                game_def.initial = initial_states[i%len(initial_states)]
            new_net = best_net.copy()
            
            #Training new net
            log.info("Training net with {} examples".format(len(training_examples)))
            new_net.train(training_examples)
            
            #Comparing nets
            log.info("Comparing networks...")
            p_old = AlphaZero(game_def,"training_old","a",best_net)
            p_new = AlphaZero(game_def,"training_new","a",new_net)
            benchmarks = Match.vs(game_def,args.n_vs,[[p_old,p_new],[p_new,p_old]],initial_states,["old_net","new_net"],penalize_illegal=args.penalize_illegal)
            log.info(benchmarks)
            new_wins = benchmarks["b"]["wins"]
            old_wins = benchmarks["a"]["wins"]
            log.info("New: Wan {}  Lost Illegal {}\nOld network: Wan {} Lost Illegal {}".format(
                new_wins,benchmarks["b"]["matches_lost_by_illegal"],
                old_wins,benchmarks["a"]["matches_lost_by_illegal"]
                ))
           
            #Updating best net           
            if new_wins > old_wins:
                log.info("{}--------------- New network is better {}vs{}------------------{}".format(bcolors.FAIL,new_wins,old_wins,bcolors.ENDC))
                best_net = new_net
                best_net.save_model(model_name="{}/{}".format(best_net.model_name,i))
                if args.vis_tree:
                    # Visualizing tree of best net
                    game_def.initial=initial_states[0]
                    state = game_def.get_initial_state()
                    p_new.visualize_net(state,"train-{}-iter-{}-new".format(new_net.model_name,i))


        log.info("Saving model")
        best_net.save_model()

    def visualize_net(self, state,name=None):
        """
        Visualizes the network of the player
        """
        tree = TreeNet.generate_from(self.game_def,self.net,state)
        name = self.name if name is None else name
        tree.print_in_file("{}/{}.png".format(self.game_def.name, name))

    def show_info(self, initial_states, args):
        """
        Shows the information for a loaded player
        """
        self.game_def.initial=initial_states[args.num_repetitions%len(initial_states)]
        state = self.game_def.get_initial_state()
        self.visualize_net(state)

    def choose_action(self,state,time_step=None,penalize_illegal=False):
        """
        The player chooses an action given a current state.

        Args:
            state (State): The current state
        
        Returns:
            action (Action): The selected action. Should be one from the list of state.legal_actions
        """
        p = state.control
        legal_actions_masked = self.game_def.encoder.mask_legal_actions(state)
        pi, v = self.net.predict_state(state)
        
        
        best_idx = np.argmax(pi)

        # Require best prediction to be legal
        if(legal_actions_masked[best_idx]==0 and penalize_illegal):
            raise IllegalActionError("Invalid action",str(self.game_def.encoder.all_actions[best_idx]))

        # Check best prediction from all legal
        legal_actions_pi = legal_actions_masked*pi
        if np.sum(legal_actions_pi)==0:
            log.info("All legal actions were predicted with 0 by {} choosing random".format(self.name))
            best_idx = randint(0,len(state.legal_actions)-1)
            legal_action = state.legal_actions[best_idx]
        else:
            best_idx = np.argmax(legal_actions_pi)
            best_name = self.game_def.encoder.all_actions[best_idx]
            legal_action = state.get_legal_action_from_str(str(best_name))
        return legal_action

