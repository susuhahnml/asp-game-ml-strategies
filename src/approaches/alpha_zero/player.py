import random
from structures.players import Player
from py_utils.logger import log
from py_utils.colors import bcolors, paint
from structures.players import Player
from py_utils.logger import log
from py_utils.colors import bcolors, paint
from structures.players import Player
from py_utils.train_utils import load_model_from_name, save_model
import numpy as np
from approaches.alpha_zero.alpha_utils import train, get_architecture,copy_model, predict
from structures.match import Match
from approaches.alpha_zero.treeZero import TreeZero
from structures.step import Step

class AlphaZero(Player):

    """
    A player using the AlphaZero algorithm

    Attributes
    ----------
    game_def: The game definition
    main_player: The name of the player to optimize
    """

    description = "A player trained using alpha zero"

    def __init__(self, game_def, name_style, main_player, model=None):
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
        name = "AlphaZero player loaded from {}".format(name_style)
        super().__init__(game_def, name, main_player)
        self.model = model if not model is None else load_model_from_name(name_style[11:],game_def.name,"alpha_zero")


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
        approach_parser.add_argument("--architecture", type=str, default="dense",
                            help="underlying neural network architecture;" +
                            " Available: 'dense', 'dense-deep', 'dense-wide', 'resnet-50'")
        approach_parser.add_argument("--n-train", type=int, default=100,
                            help="Number of times the network will be trained and tested")
        approach_parser.add_argument("--n-episodes", type=int, default=5,
                            help="Number episodes computed with MCTS to get training examples")
        approach_parser.add_argument("--n-epochs", type=int, default=900,
                            help="Epochs for each training")
        approach_parser.add_argument("--batch-size", type=int, default=200,
                            help="Batch size for each training")
        approach_parser.add_argument("--lr", type=float, default=0.01,
                            help="Learning rate for training")
        approach_parser.add_argument("--n-vs", type=float, default=30,
                            help="Number of matches to compare networks")
        approach_parser.add_argument("--n-mcts-simulations", type=float, default=100,
            help="Number of times the MCTS algorithm will transverse to compute probabilities")
        approach_parser.add_argument("--model-name", type=str, default="unnamed",
                            help="Name of the model, used for saving and logging")
        approach_parser.add_argument("--rand-init", type=int, default=None,
        help="Random seed for creating initial states in training net")


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
        best_net = get_architecture(game_def,args.lr,architecture_name=args.architecture)
        game_def.get_random_initial()
        using_random = not args.rand_init is None
        if(using_random):
            log.info("Using random seed {} for initial states in training".format(args.rand_init))
            game_def.get_random_initial()
            initial_states = game_def.random_init
            random.Random(args.rand_init).shuffle(initial_states)
        else:
            log.info("Using default initial state in training {} ".format(game_def.initial))
            initial_states = [game_def.initial]

        for i in range(args.n_train):
            log.info("Iteration {}...".format(i))
            training_examples = []
            for e in range(args.n_episodes):
                log.info("\t\tEpisode {}...".format(e))
                new_examples = TreeZero.run_episode(game_def, best_net, args,e)
                training_examples+=new_examples
                game_def.initial = initial_states[i%len(initial_states)]
            new_net = copy_model(game_def,best_net,args.lr)
            log.info("Training net with {} examples".format(len(training_examples)))
            train(new_net,training_examples,args.n_epochs,args.batch_size)

            log.info("Comparing networks...")
            p_old = AlphaZero(game_def,"training_old","a",best_net)
            p_new = AlphaZero(game_def,"training_new","a",new_net)
            benchmarks = Match.vs(game_def,args.n_vs,[[p_old,p_new],[p_new,p_old]],initial_states,["old_net","new_net"])
            if benchmarks["b"]["wins"] > benchmarks["a"]["wins"]:
                log.info("{}--------------- New network is better {}vs{}------------------{}".format(bcolors.FAIL,benchmarks["b"]["wins"],benchmarks["a"]["wins"],bcolors.ENDC))
                best_net = new_net

        log.info("Saving model")
        save_model(best_net,args.model_name,game_def.name,"alpha_zero")

        
    def choose_action(self,state):
        """
        The player chooses an action given a current state.

        Args:
            state (State): The current state
        
        Returns:
            action (Action): The selected action. Should be one from the list of state.legal_actions
        """
        p = state.control
        possible_actions_masked = self.game_def.encoder.mask_legal_actions(state)
        pi, v = predict(self.game_def,state,self.model)
        
        best_idx = np.argmax(possible_actions_masked*pi)
        best_name = self.game_def.encoder.all_actions[best_idx]
        
        legal_action = state.get_legal_action_from_str(str(best_name))
        # log.info("Best prediction of {} with proba {}: \n{}".format(self.name,round(pi[best_idx],2),Step(state,legal_action,0).ascii))
        return legal_action
