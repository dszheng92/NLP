""" Maximum entropy model for Assignment 1: Starter code.

You can change this code however you like. This is just for inspiration.

"""
import numpy as np
import propername
import scipy.optimize as opt
from propername import Propername_Feature_Extract
import pandas as pd
#from util import evaluate, load_data
import pandas as pd


# tune 1 iter 100 nrange 14 0.85
# tune 1 iter 100 nrange 13 0.841
# tune 1 iter 50 nrange 12 0.779
# tune 1 iter 100 nrange 12 0.783
# tune 1 iter 150 nrange 12 0.788

class MaximumEntropyModel():
    """ Maximum entropy model for classification.

    Attributes:

    """
    def __init__(self):
        self.N = 0
        self.num_x = 0 # feature number
        self.W = []
        self.num_y = 0 # class number
        self.tune=1

    def train(self, X_train, Y_train):
        self.N = X_train.shape[0]
        self.num_y = max(Y_train)+1
        self.num_x = X_train.shape[1]
        self.W = np.zeros(self.num_y*self.num_x)
        self.W,f,d = opt.fmin_l_bfgs_b(func=self.cal_log, x0=self.W, fprime=self.cal_log_gradient, approx_grad=False, args=(X_train, Y_train), iprint=1, maxiter=100)


    def cal_log(self,*args):
        W, X_train, Y_train = args
        ws = W.reshape(self.num_y,self.num_x)
        pw = np.dot(X_train,ws.T)
        pyx = self.cal_prb(pw) # each feature's score on each yi's weight x1y1,x1y2,x1y3...
        pyixi = []
        for i in range(self.N):
            pyixi.append(pyx[i][Y_train[i]])
        logyx = np.sum(np.log(pyixi))-0.5*self.tune*np.sum(ws**2)  # maximize it
        return -logyx

    def cal_log_gradient(self,*args):
        W, X_train, Y_train= args
        ws = W.reshape(self.num_y, self.num_x)
        logyx_grad = np.zeros((ws.shape[0],ws.shape[1]))
        pw = np.dot(X_train, ws.T)
        pyx = self.cal_prb(pw)
        for i in range(self.num_y):
            logyx_grad[i]+=np.sum(X_train[Y_train==i],axis=0)
            logyx_grad[i]-=np.sum(X_train*pyx[:,i].reshape(-1,1),axis=0)
        logyx_grad -= self.tune*ws
        return -logyx_grad.flatten()

    def cal_prb(self,wf):
        wf = np.exp(wf) / np.sum(np.exp(wf),axis=1).reshape(-1,1)
        return wf

    def predict(self, model_input):
        ws = self.W.reshape(self.num_y,self.num_x)
        pred_y = np.argmax(np.dot(model_input,ws.T),axis=1)
        return pred_y


if __name__ == "__main__":
    #train_data, dev_data, test_data, data_type = load_data(sys.argv)

    propername = Propername_Feature_Extract()
    train_data, train_labels, dev_data, dev_labels, test_data = propername.propername_data_loader(
        "train/train_data.csv",
        "train/train_labels.csv",
        "dev/dev_data.csv", "dev/dev_labels.csv",
        "test/test_data.csv")
    X_train, X_dev, X_test = propername.propername_featurize(train_data, dev_data, test_data, ngram_range=(1, 4))
    Y_train, Y_dev = propername.propername_label_to_id(train_labels, dev_labels)


    #Y_train = propername.propername_id_to_vector(Y_train)

    # Train the model using the training data.
    model = MaximumEntropyModel()
    #print(X_train.shape,Y_train.shape)
    model.train(X_train,Y_train)
    Y_dev_predicted = model.predict(X_dev)
    # print(Y_dev_predicted[:100])
    score = np.sum(Y_dev == Y_dev_predicted) / Y_dev.shape[0]
    print(score)
    # Y_test = model.predict(X_test)
    # Y_test_label = propername.propername_id_to_label(Y_test.tolist())
    # res_se = pd.Series(Y_test_label)
    # res_df = res_se.to_frame()
    # res_df.to_csv("res.csv")

