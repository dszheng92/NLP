""" Contains the Word2VecModel class, which can be trained, and also used to
    evaluate word similarity betwen two pairs of words. 

    This is starter code. Feel free to change this code however you like.
"""

def load_training_data():
    """ Loads training data from the 1m corpus (in the data/training directory).
        This is where you should also do featurization, extract contexts, etc."
    """
    # TODO: implement data loading.
    pass

def load_eval_data()
    """ Loads evaluation data from the data/similarity directory. You should load
        the development and test pairs, as well as the development labels (scores).
    """
    # TODO: implement loading of evaluation data.
    pass

def evaluate(predicted_similarities, human_ratings):
    """ Computes a score how similar the predicted similarities were to the human
        ratings.

        The implementation should be similar to that in evaluation.py.
    """
    # TODO: implement evaluation function
    pass

class Word2VecModel:
    """ Word2vec model, that maps a word to a vector representation.

    Attributes:
    """

    def __init__(self):
        """Learns embeddings for words given text data.
        """
        # TODO: implement this function.
        pass

    def train(self, data):
        """ Uses the data to train a word embedder.

        Inputs:
            data: Any type, depending on how you plan to train the model or what
                kind of featurization you use.
        """
        # TODO: implement model training.
        pass

    def predict_similarity(self, word1, word2):
        """ Predicts a similarity score between two words according to the model's
            embeddings.

            The implementation should be similar to that in similarity.py.
            
        Inputs:
            word1: Any type (string, integer) representing one of the words to try.
            word2: Any type (string, integer), representing the other word to try.
        """
        # TODO: implement pairwise evaluation.
        pass

    def save_embeddings(self):
        """ Saves embeddings so that they can be used by the similarity/evaluate
            scripts to predict similarities between words.
        """
        # TODO: implement embedding saving.

if __name__ == "__main__":
    training_data = load_training_data()
    model = Word2VecModel()

    # Fit the model to the training corpus and save the parameters.
    model.train(training_data)
    model.save_embeddings()
    

    # Evaluate on the dev data.
    # We recommend that you use the similarity.py and evaluate.py scripts
    # to generate similarity scores for the test data, instead of running it here.
    dev_pairs, dev_labels, _ = load_eval_data()

    predicted_similarities = [model.predict_similarity(pair) for pair in dev_pairs]

    print("Dev: " + str(evaluate(predicted_similarities, dev_labels)))
