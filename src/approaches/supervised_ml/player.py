from structures.players import Player
from py_utils.logger import log
from py_utils.colors import bcolors, paint
from structures.players import Player, IllegalActionError
from py_utils.logger import log
from py_utils.colors import bcolors, paint
from structures.players import Player
import numpy as np
from approaches.supervised_ml.net_supervised import NetSupervised
from structures.tree_net import TreeNet
from random import randint


class MLPlayer(Player):

    """
    A Supervised Machine learning player

    Attributes
    ----------
    game_def: The game definition
    main_player: The name of the player to optimize
    """

    description = "A player trained using supervided learning"

    def __init__(self, game_def, name_style, main_player):
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
        name = "Supervised Machine Learning player loaded from {}".format(name_style)
        super().__init__(game_def, name, main_player)
        model_name = name_style[14:]
        self.net = NetSupervised(game_def,model_name)
        self.net.load_model_from_file()


    @classmethod
    def get_name_style_description(cls):
        """
        Gets a description of the required format of the name_style.
        This description will be shown on the console.
        Returns: 
            String for the description
        """
        return "supervised_ml-<file-name> where file-name indicates the name of the saved model inside ml_agent/saved_models/game_name"

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
        return name_style[:14]=="supervised_ml-"


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
                            help="underlying neural network architecture;" +
                            " Available: 'default'")
        approach_parser.add_argument("--n-epochs", type=int, default=50000,
                            help="total number of steps to take in environment")
        approach_parser.add_argument("--model-name", type=str, default="unnamed",
                            help="name of the model, used for saving and logging")
        approach_parser.add_argument("--training-file", type=str, required=True,
                            help="Name of the file staring in train/game_name")


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
        net = NetSupervised(game_def,args.model_name,model=None,args=args)
        net.load_model_from_args()
        net.train()
        net.save_model()

        
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
        pi = self.net.predict_state(state)
        
        
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