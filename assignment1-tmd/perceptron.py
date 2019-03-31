
import os
import sys
import numpy as np
import propername
from propername import Propername_Feature_Extract
import pandas as pd
#from util import evaluate, load_data
# lr 0.01 iter 100 nrange(1,1) 0.4991
# lr 0.01 iter 100 nrange(1,2) 0.7369
# lr 0.01 iter 100 nrange(1,3) 0.8012
# lr 0.01 iter 100 nrange(1,4) 0.8524
# lr 0.01 iter 500 nrange(1,2) 0.7069
# lr 0.001 iter 100 nrange(1,2) 0.7300
# lr 0.001 iter 500 nrange(1,2) 0.7245
class PerceptronModel():
    """ Maximum entropy model for classification.

    Attributes:

    """
    def __init__(self,lr=0.01, n_iter=100):
        self.lr = lr
        self.n_iter = n_iter

    def train(self, X_train, Y_train):
        self.weight = np.zeros((1+X_train.shape[1], Y_train.shape[1]))
        X_train = np.hstack((np.ones((X_train.shape[0],1)), X_train))
        # for each dimension
        for i in range(self.n_iter):
            if i%10==0: print("iteration ",i)
            # print(i)
            for x,y in zip(X_train, Y_train):
                y_pred = np.argmax(np.dot(x, self.weight))
                if y[y_pred]!=1:
                    idx = np.where(y==1)[0][0]
                    self.weight[:, y_pred] -= self.lr * x
                    self.weight[:, idx] += self.lr * x

        return self


    def predict(self, model_input):
        model_input = np.hstack((np.ones((model_input.shape[0], 1)), model_input))
        return np.argmax(np.dot(model_input,self.weight),axis=1)

    def error_analysis(self,Y_dev,Y_dev_predicted,dev_data,propername):
        index = [i for i in range(Y_dev.shape[0])]
        # build the confusion matrix
        confusion_matrix=[[0 for i in range(5)]for j in range(5)]
        for i in index:
            confusion_matrix[Y_dev[i]][Y_dev_predicted[i]]+=1
        print(np.array(confusion_matrix))
        print(np.array(confusion_matrix)/np.sum(np.array(confusion_matrix)))
        # print some error examples
        selected = np.where(Y_dev!=Y_dev_predicted)[0]
        selected = selected[:20]
        for i in selected:
            print("example ",i)
            print(dev_data[i])
            print("actual: ",propername.id_to_class[Y_dev[i]],"predicted:",propername.id_to_class[Y_dev_predicted[i]])


if __name__ == "__main__":
    #train_data, dev_data, test_data, data_type = load_data(sys.argv)

    propername = Propername_Feature_Extract()
    train_data, train_labels, dev_data, dev_labels, test_data = propername.propername_data_loader(
        "train/train_data.csv",
        "train/train_labels.csv",
        "dev/dev_data.csv", "dev/dev_labels.csv",
        "test/test_data.csv")
    X_train, X_dev, X_test = propername.propername_featurize(train_data, dev_data, test_data, ngram_range=(1, 2))
    Y_train, Y_dev = propername.propername_label_to_id(train_labels, dev_labels)
    Y_train = propername.propername_id_to_vector(Y_train)

    # Train the model using the training data.
    model = PerceptronModel()
    model.train(X_train,Y_train)

    Y_dev_predicted = model.predict(X_dev)

    score = np.sum(Y_dev == Y_dev_predicted) / Y_dev.shape[0]
    print(score)

    model.error_analysis(Y_dev,Y_dev_predicted,dev_data,propername)
    # Y_test = model.predict(X_test)
    # Y_test_label = propername.propername_id_to_label(Y_test.tolist())
    # res_se = pd.Series(Y_test_label)
    # res_df = res_se.to_frame()
    # res_df.to_csv("iter100lr01n14.csv")



