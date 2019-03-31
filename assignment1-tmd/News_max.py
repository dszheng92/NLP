import numpy as np
import newsgroup
import scipy.optimize as opt
from newsgroup import Newsgroup_Feature_Extract

#from util import evaluate, load_data
import pandas as pd

class MaximumEntropyModel():
    """ Maximum entropy model for classification.

    Attributes:

    """
    def __init__(self):
        self.N = 0
        self.num_x = 0 # feature number
        self.W = []
        self.num_y = 0 # class number

    def train(self, X_train, Y_train, tune = 1):
        self.N = X_train.shape[0]
        self.num_y = max(Y_train) + 1
        self.num_x = X_train.shape[1]
        self.W = np.zeros(self.num_y*self.num_x)
        self.W,f,d = opt.fmin_l_bfgs_b(func=self.cal_log, x0=self.W, fprime=self.cal_log_gradient, approx_grad=False, args=(X_train, Y_train, tune), iprint=1, maxiter=130)

    def cal_log(self,*args):
        W, X_train, Y_train, tune = args
        ws = W.reshape(self.num_y,self.num_x)
        pw = np.dot(X_train,ws.T)
        pyx = self.cal_prb(pw) # each feature's score on each yi's weight x1y1,x1y2,x1y3...
        pyixi = []
        for i in range(self.N):
            pyixi.append(pyx[i][Y_train[i]])
        logyx = np.sum(np.log(pyixi))-0.5*tune*np.sum(ws**2)  # maximize it
        return -logyx

    def cal_log_gradient(self,*args):
        W, X_train, Y_train, tune = args
        ws = W.reshape(self.num_y, self.num_x)
        logyx_grad = np.zeros((ws.shape[0],ws.shape[1]))
        pw = np.dot(X_train, ws.T)
        pyx = self.cal_prb(pw)
        for i in range(self.num_y):
            logyx_grad[i]+=np.sum(X_train[Y_train==i],axis=0)
            logyx_grad[i]-=np.sum(X_train*pyx[:,i].reshape(-1,1),axis=0)
        logyx_grad -= ws
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

    newsData = Newsgroup_Feature_Extract()
    X_train, Y_train, X_dev, Y_dev, X_test = newsData.bag_of_words(ngram_range=(1, 2), dim_used=30000)
    #Y_train = newsData.newsgroup_id_to_vector(Y_train)

    # Train the model using the training data.
    model = MaximumEntropyModel()
    #print(X_train.shape,Y_train.shape)
    model.train(X_train, Y_train)
    Y_dev_predicted = model.predict(X_dev)
    # print(Y_dev_predicted[:100])
    score = np.sum(Y_dev == Y_dev_predicted) / Y_dev.shape[0]
    print(score)
    Y_test = model.predict(X_test)
    Y_test_label = newsData.newsgroup_id_to_label(Y_test.tolist())
    res_se = pd.Series(Y_test_label)
    res_df = res_se.to_frame()
    res_df.to_csv("130max_iter100lr07.csv")

