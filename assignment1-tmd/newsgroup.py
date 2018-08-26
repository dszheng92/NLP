#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import csv
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import re
import sys


#This is for feature extraction of 20 Newsgroups.
csv.field_size_limit(sys.maxsize)
stopWords = set(stopwords.words('english'))
stemmer = PorterStemmer()


class Newsgroup_Feature_Extract:
    def __init__(self):
        self.data_path = '../assignment1-tmd/data/newsgroups/'
        self.vocab_data = None
        self.vocab_label = None
        self.class_dict = None


    def bag_of_words(self, ngram_range = (1, 1), dim_used = None):
        texts_train, Y_train = self._import_data('train')
        texts_dev, Y_dev = self._import_data('dev')
        texts_test, _ = self._import_data('test')
        vectorizer = CountVectorizer(max_features = dim_used, ngram_range = ngram_range, stop_words = stopWords, tokenizer = self._stem_tokenize, binary = True)
        X_train = vectorizer.fit_transform(texts_train).toarray()
        X_dev = vectorizer.transform(texts_dev).toarray()
        X_test = vectorizer.transform(texts_test).toarray()
        self.vocab_data = vectorizer.vocabulary_
        #dictionary = self.class_dict
        #labels = self.vocab_label
        return X_train, Y_train, X_dev, Y_dev, X_test


    def _import_data(self, scope):
        data_path = self.data_path
        articles = []
        with open(data_path + scope + '/' + scope + '_data.csv') as dataLines:
            next(dataLines)
            reader = csv.reader(dataLines)
            for m in reader:
                m = m[1]
                articles.append(m)
        articles = np.array(articles)

        if scope == 'test':
            target = None
        elif scope == 'train':
            labels = []
            with open(data_path + scope + '/' + scope + '_labels.csv') as dataLabels:
                next(dataLabels)
                reader = csv.reader(dataLabels)
                for line in reader:
                    labels.append(line[1])
            labels = np.array(labels)

            #Class dictionary is the disctionary of number of classes. It is the opposite of labels dictionary
            #Which is to categorize classes with numbers.

            vocab_label = {classes: num for classes, num in zip(list(set(labels)), range(len(labels)))}
            self.vocab_label = vocab_label
            class_dict = {classes: num for classes, num in zip(range(len(labels)), list(set(labels)))}
            self.class_dict = class_dict

            target = np.zeros_like(labels)
            for i in range(len(labels)):
                target[i] = vocab_label[labels[i]]
            target = target.astype(int)
        elif scope == 'dev':
            vocab_label = self.vocab_label
            labels = []
            with open(data_path + scope + '/' + scope + '_labels.csv') as f_label:
                next(f_label)
                reader = csv.reader(f_label)
                for line in reader:
                    labels.append(line[1])
            labels = np.array(labels)
            target = np.zeros_like(labels)
            for i in range(len(labels)):
                target[i] = vocab_label[labels[i]]
            target = target.astype(int)
        return articles, target


    def _stem_tokenize(self, art):
        stemTokens = [stemmer.stem(n) for n in word_tokenize(art) if n.isalpha()]
        return stemTokens

    def newsgroup_id_to_label(self, Y_id):
        Y_test=[]
        for i in range(len(Y_id)):
            Y_test.append(self.class_dict[Y_id[i]])
        return Y_test

    def newsgroup_id_to_vector(self, Y_id):
        Y_train = np.zeros((Y_id.shape[0], 20))
        for i in range(Y_id.shape[0]):
            Y_train[i,Y_id[i]]=1
        return Y_train


if __name__ == '__main__':

    newsData = Newsgroup_Feature_Extract()
    X_train, Y_train, X_dev, Y_dev, X_test = newsData.bag_of_words(ngram_range=(1, 2), dim_used=5000)

    #print('dictionary', dictionary, 'labels',  labels )
    #print(X_train.shape)
   # print("Y before:", Y_train)
    #Y_train = newsData.newsgroup_id_to_vector(Y_train)
    #print('Y_train:', Y_train)




