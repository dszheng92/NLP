import os
import sys
#import dynet_config
#dynet_config.set_gpu()
import dynet as dy
import numpy as np
import propername
from propername import Propername_Feature_Extract
import pandas as pd
import datetime
#from util import evaluate

dy.renew_cg()


class MultilayerPerceptronModel():

    def __init__(self, m, in_dim, hid_dim, out_dim, non_lin=dy.tanh, opt=dy.AdamTrainer, n_iter=10):
        self._pw1 = m.add_parameters((in_dim, hid_dim), init=dy.GlorotInitializer())
        self._pw2 = m.add_parameters((hid_dim, hid_dim), init=dy.GlorotInitializer())
        self._pw3 = m.add_parameters((hid_dim, out_dim), init=dy.GlorotInitializer())
        self._pb1 = m.add_parameters((hid_dim,), init=dy.GlorotInitializer())
        self._pb2 = m.add_parameters((hid_dim,), init=dy.GlorotInitializer())
        self._pb3 = m.add_parameters((out_dim,), init=dy.GlorotInitializer())
        self.non_lin = non_lin
        self.opt = opt(m)
        self.n_iter = n_iter

    def _multilayer_perceptron(self, x):

        g = self.non_lin

        layer_1 = g(dy.transpose(dy.transpose(x * self.weights['h1']) + self.biases['b1']))

        layer_2 = g(dy.transpose(dy.transpose(layer_1 * self.weights['h2']) + self.biases['b2']))

        out_layer = dy.softmax(dy.transpose(layer_2 * self.weights['out']) + self.biases['out'])

        return out_layer

    def _create_network(self, inputs, expected_answer):
        dy.renew_cg()  # new computation graph

        self.weights = {
            'h1': dy.parameter(self._pw1),
            'h2': dy.parameter(self._pw2),
            'out': dy.parameter(self._pw3)
        }

        self.biases = {
            'b1': dy.parameter(self._pb1),
            'b2': dy.parameter(self._pb2),
            'out': dy.parameter(self._pb3)
        }

        x = dy.vecInput(len(inputs))
        # x = dy.vecInput(len(X_train[1]))
        x.set(inputs)
        x = dy.reshape(x, (1, X_train.shape[1]))
        y = dy.inputTensor(expected_answer)
        yy = dy.reshape(y, (5, 1))
        output = self._multilayer_perceptron(x)
        loss = dy.binary_log_loss(output, yy)
        return loss

    def train(self, A, B):
        """

        Inputs:
            training_data: Suggested type is (list of pair), where each item is
                a training example represented as an (input, label) pair.
        """

        seen_instances = total_loss = cumiter = 0
        # close = 0
        for kk in range(self.n_iter):
            print('iteration', kk)
            for x, y in zip(A, B):
                loss = self._create_network(x, y)
                seen_instances += 1
                cumiter += 1
                total_loss += loss.value()
                loss.backward()
                self.opt.update()
                if (seen_instances > 1 and seen_instances % 200 == 0):
                    print("average loss is:", total_loss / seen_instances, 'running', (cumiter / 200) / (kk + 1),
                          'iter:', kk)
                    print(datetime.datetime.now().time())
                    seen_instances = total_loss = 0

    def predict(self, inputy):
        """ Predicts a label for an input.

        Inputs:
            model_input (features): Input data for an example, represented as a
                feature vector.

        Returns:
            The predicted class.

        """
        accu = self._multilayer_perceptron(dy.inputTensor(inputy))

        # score = np.sum(label == np.argmax(accu.npvalue(), axis=0)) / Y_dev.shape[0]

        return accu


if __name__ == "__main__":

    propername = Propername_Feature_Extract()
    train_data, train_labels, dev_data, dev_labels, test_data = propername.propername_data_loader(
        "train/train_data.csv",
        "train/train_labels.csv",
        "dev/dev_data.csv", "dev/dev_labels.csv",
        "test/test_data.csv")
    X_train, X_dev, X_test = propername.propername_featurize(train_data, dev_data, test_data, ngram_range=(1, 4))
    Y_train, Y_dev = propername.propername_label_to_id(train_labels, dev_labels)
    Y_train1 = propername.propername_id_to_vector(Y_train)

    dy.renew_cg()

    #n_hidden = 2048
    n_hidden = 64
    n_input = X_train.shape[1]
    n_classes = Y_train1.shape[1]

    m = dy.ParameterCollection()
    model = MultilayerPerceptronModel(m, n_input, n_hidden, n_classes, dy.tanh, dy.AdamTrainer, 10)
    #training_data = zip(X_train, Y_train1)
    #dev = zip(X_dev,Y_dev)
    model.train(X_train, Y_train1)
    Y_dev_predicted = model.predict(X_dev)
    score = np.sum(Y_dev == np.argmax(Y_dev_predicted.npvalue(), axis=0)) / Y_dev.shape[0]
    print(score)
    Y_test = model.predict(X_test)
    Y_test1 = np.argmax(Y_test.npvalue(), axis=0)
    Y_test_label = propername.propername_id_to_label(Y_test1.tolist())
    res_se = pd.Series(Y_test_label)
    res_df = res_se.to_frame()
    res_df.to_csv("propmlp_iter100lr5prop.csv")