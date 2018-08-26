import json

from argparse import ArgumentParser
from model import Model

def load_data(filename):
    """Loads the data from the JSON files.

    You are welcome to create your own class storing the data in it; e.g., you
    could create AlchemyWorldStates for each example in your data and store it.

    Inputs:
        filename (str): Filename of a JSON encoded file containing the data. 

    Returns:
        examples
    """
    pass

def train(model, train_data):
    """Finds parameters in the model given the training data.

    TODO: implement this function -- suggested implementation iterates over epochs,
        computing loss over training set (in batches, maybe), evaluates on a held-out set
        at each round (you are welcome to split train_data here, or elsewhere), and
        saves the final model parameters.

    Inputs:
        model (Model): The model to train.
        train_data (list of examples): The training examples given.
    """
    pass

def execute(world_state, action_sequence):
    """Executes an action sequence on a world state.

    TODO: This code assumes the world state is a string. However, you may sometimes
    start with an AlchemyWorldState object. I suggest loading the AlchemyWorldState objects
    into memory in load_data, and moving that part of the code to load_data. The following
    code just serves as an example of how to 1) make an AlchemyWorldState and 2) execute
    a sequence of actions on it.

    Inputs:
        world_state (str): String representing an AlchemyWorldState.
        action_sequence (list of str): Sequence of actions in the format ["action arg1 arg2",...]
            (like in the JSON file).
    """
    alchemy_world_state = AlchemyWorldState(world_state)
    fsa = AlchemyFSA(alchemy_world_state)

    for action in action_sequence:
        split = action.split(" ")
        act = split[0]
        arg1 = split[1]
        
        # JSON file doesn't contain  NO_ARG.
        if len(split) < 3:
            arg2 = NO_ARG
        else:
            arg2 = split[2]

        fsa.feed_complete_action(act, arg1, arg2)

    return fsa.world_state()

def predict(model, data, outname):
    """Makes predictions for data given a saved model.

    This function should predict actions (and call the AlchemyFSA to execute them),
    and save the resulting world states in the CSV files (same format as *.csv).

    TODO: you should implement both "gold-previous" and "entire-interaction"
        prediction.

    In the first case ("gold-previous"), for each utterance, you start in the previous gold world state,
    rather than the on you would have predicted (if the utterance is not the first one).
    This is useful for analyzing errors without the problem of error propagation,
    and you should do this first during development on the dev data.

    In the second case ("entire-interaction"), you are continually updating
    a single world state throughout the interaction; i.e. for each instruction, you start
    in whatever previous world state you ended up in with your prediction. This method can be
    impacted by cascading errors -- if you made a previous incorrect prediction, it's hard
    to recover (or your instruction might not make sense). 

    For test labels, you are expected to predict /final/ world states at the end of each
    interaction using "entire-interaction" prediction (you aren't provided the gold
    intermediate world states for the test data).

    Inputs:
        model (Model): A seq2seq model for this task.
        data (list of examples): The data you wish to predict for.
        outname (str): The filename to save the predictions to.
    """
    pass

def main():
    # A few command line arguments
    parser = ArgumentParser()
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--predict", type=bool, default=False)
    parser.add_argument("--saved_model", type=str, default="")
    args = parser.parse_args()

    assert args.train or args.predict

    # Load the data; you can also use this to construct vocabularies, etc.
    train_data = load_data("train.json")
    dev_data = load_data("dev.json")

    # Construct a model object.
    model = Model()

    if args.train:
       # Trains the model
       train(model, train_data) 
    if args.predict:
        # Makes predictions for the data, saving it in the CSV format
        assert args.saved_model

        # TODO: you can modify this to take in a specified split of the data,
        # rather than just the dev data.
        predict(model, dev_data)

        # Once you predict, you can run evaluate.py to get the instruction-level
        # or interaction-level accuracies.

if __name__ == "__main__":
    main()
