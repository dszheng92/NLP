import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import sys


class Propername_Feature_Extract:

    def __init__(self):
        self.path = "data/propernames/"
        self.class_to_id = {'place': 0, 'person': 1, 'drug': 2, 'company': 3, 'movie': 4}
        self.id_to_class = {0: 'place', 1: 'person', 2: 'drug', 3: 'company', 4: 'movie'}

    def propername_featurize(self,train_data,dev_data,test_data,ngram_range=(2, 2)):
        vectorizer = CountVectorizer(analyzer='char', ngram_range=ngram_range)
        X_train = vectorizer.fit_transform(train_data.flatten()).toarray()
        X_dev = vectorizer.transform(dev_data.flatten()).toarray()
        X_test = vectorizer.transform(test_data.flatten()).toarray()
        return X_train,X_dev,X_test

    def propername_label_to_id(self,train_labels,dev_labels):
        Y_train=[]
        Y_dev = []
        for i in range(len(train_labels)):
            Y_train.append(self.class_to_id[train_labels[i,0]])
        for i in range(len(dev_labels)):
            Y_dev.append(self.class_to_id[dev_labels[i,0]])
        return np.array(Y_train),np.array(Y_dev)

    def propername_id_to_label(self,Y_id):
        Y_test=[]
        for i in range(len(Y_id)):
            Y_test.append(self.id_to_class[Y_id[i]])
        return Y_test

    def propername_id_to_vector(self,Y_id):
        Y_train = np.zeros((Y_id.shape[0],5))
        for i in range(Y_id.shape[0]):
            Y_train[i,Y_id[i]]=1
        return Y_train

    def propername_data_loader(self,train_data_filename,
                               train_labels_filename,
                               dev_data_filename,
                               dev_labels_filename,
                               test_data_filename):
        """ Loads the data.

        Inputs:
            train_data_filename (str): The filename of the training data.
            train_labels_filename (str): The filename of the training labels.
            dev_data_filename (str): The filename of the development data.
            dev_labels_filename (str): The filename of the development labels.
            test_data_filename (str): The filename of the test data.

        Returns:
            Training, dev, and test data, all represented as (input, label) format.

            Suggested: for test data, put in some dummy value as the label.
        """
        train_data = pd.read_csv(self.path+train_data_filename).as_matrix()[:,1:]
        train_labels = pd.read_csv(self.path+train_labels_filename).as_matrix()[:,1:]
        dev_data = pd.read_csv(self.path + dev_data_filename).as_matrix()[:,1:]
        dev_labels = pd.read_csv(self.path + dev_labels_filename).as_matrix()[:,1:]
        test_data = pd.read_csv(self.path + test_data_filename).as_matrix()[:,1:]

        return train_data,train_labels,dev_data,dev_labels,test_data

if __name__ == '__main__':
    propername = Propername_Feature_Extract()
    train_data, train_labels, dev_data, dev_labels, test_data = propername.propername_data_loader("train/train_data.csv",
                                                                               "train/train_labels.csv",
                                                                               "dev/dev_data.csv", "dev/dev_labels.csv",
                                                                               "test/test_data.csv")
    X_train,  X_dev,  X_test = propername.propername_featurize(train_data,dev_data,test_data,ngram_range=(1, 2))
    Y_train,  Y_dev = propername.propername_label_to_id(train_labels,dev_labels)
    print(Y_train.shape)
    Y_train = propername.propername_id_to_vector(Y_train)
    print(X_train.shape)
    print(Y_train.shape)
    print('train_labels:', train_labels)
    print('Y_dev:', dev_labels)
    print('Y_train:', Y_train)
    print('Y_dev:', Y_dev)

