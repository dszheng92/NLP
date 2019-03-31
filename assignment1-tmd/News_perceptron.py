import os
import sys
import numpy as np
import newsgroup
from newsgroup import Newsgroup_Feature_Extract
import pandas as pd
#from util import evaluate, load_data

class PerceptronModel():
    """ Perceptron model for 20 Newsgroup classification.

    Attributes:

    """
    def __init__(self,lr=0.01, n_iter=300):
        self.lr = lr
        self.n_iter = n_iter

    def train(self, X_train,Y_train):
        self.weight = np.zeros((1+X_train.shape[1],Y_train.shape[1]))
        X_train = np.hstack((np.ones((X_train.shape[0],1)),X_train))
        # for each dimension
        for i in range(self.n_iter):
            print(i)
            for x,y in zip(X_train, Y_train):
                y_pred = np.argmax(np.dot(x, self.weight))
                if y[y_pred]!=1:
                    idx = np.where(y==1)[0][0]
                    self.weight[:, y_pred] -= self.lr*x
                    self.weight[:, idx] += self.lr*x

        return self


    def predict(self, model_input):
        model_input = np.hstack((np.ones((model_input.shape[0], 1)), model_input))
        return np.argmax(np.dot(model_input, self.weight),axis=1)

if __name__ == "__main__":
    #train_data, dev_data, test_data, data_type = load_data(sys.argv)

    newsData = Newsgroup_Feature_Extract()
    X_train, Y_train, X_dev, Y_dev, X_test = newsData.bag_of_words(ngram_range=(1, 1), dim_used=20000)
    Y_train = newsData.newsgroup_id_to_vector(Y_train)

    # Train the model using the training data.
    model = PerceptronModel()
    model.train(X_train, Y_train)

    Y_dev_predicted = model.predict(X_dev)
    score = np.sum(Y_dev == Y_dev_predicted) / Y_dev.shape[0]
    print(score)
    Y_test = model.predict(X_test)
    Y_test_label = newsData.newsgroup_id_to_label(Y_test.tolist())
    res_se = pd.Series(Y_test_label)
    res_df = res_se.to_frame()
    res_df.to_csv("11!!iter100lr07.csv")


