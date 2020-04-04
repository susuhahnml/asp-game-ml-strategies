from structures.players import Player
from py_utils.logger import log
from py_utils.colors import bcolors, paint
from structures.players import Player
from py_utils.logger import log
from py_utils.colors import bcolors, paint
from structures.players import Player
from approaches.supervised_ml.ml_utils import train
from py_utils.train_utils import load_model_from_name, save_model
import numpy as np
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
        self.model = load_model_from_name(name_style[13:],game_def.name,"supervised_ml")


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
        return name_style[:13]=="supervised_ml-"


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
        model = train(game_def,args.architecture,args.n_epochs,args.training_file)
        save_model(model,args.model_name,game_def.name,"supervised_ml")

        
    def choose_action(self,state):
        """
        The player chooses an action given a current state.

        Args:
            state (State): The current state
        
        Returns:
            action (Action): The selected action. Should be one from the list of state.legal_actions
        """
        state_masked = self.game_def.encoder.mask_state(state)
        possible_actions_masked = [(str(a.action), self.game_def.encoder.mask_action(str(a.action))) for a in state.legal_actions]
        inputs = [np.concatenate([state_masked,a]) for n,a in possible_actions_masked]
        inputs = np.array(inputs)
        predictions = self.model.predict([inputs])
        v_predictions = predictions[0]
        r_predictions = predictions[1]
        
        best = (None,-float("inf"))
        for i,(n,p) in enumerate(possible_actions_masked):
            v = v_predictions[i][0]
            r = r_predictions[i][0]
            if v>best[1]:
                best = (n,v)
            log.debug("Predictions for {} v={} r={}".format(n,v,r))
        log.debug("Best prediction: {} ".format(best))
        
        legal_action = state.get_legal_action_from_str(best[0])
        if not legal_action:
            log.error("ML Player select a non leagal action")
        return legal_action