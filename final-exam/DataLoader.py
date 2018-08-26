import numpy as np
import nltk
import os
import sys
UNK = '<unk>'
PAD = '<pad>'
NULL = 'NULL'

class DataLoader:
    def __init__(self):
        self.train_label_path = "./data/train/train.txt"
        self.dev_label_path = "./data/dev/dev.txt"
        self.test_path = "./data/test/test.nolabels.txt"
        self.word_to_id = None
        self.id_to_word = None
        self.tag_to_id = None
        self.id_to_tag = None
        self.label_to_id = None
        self.id_to_label = None
        self.max_len=52

    def load_data(self,mode):
        sentences, tags, labels = self.build_sentences(mode)
        if mode == 'train':self.build_vocab(sentences,tags,labels)
        x,pos_tags,y = self.build_data(sentences,tags,labels,mode)
        return x,pos_tags,y

    def build_data(self,sentences, tags, labels, mode):
        sentences_inputs = np.zeros((len(sentences),self.max_len)).astype(int)
        tags_inputs = np.zeros(sentences_inputs.shape).astype(int)
        labels_inputs = np.zeros(sentences_inputs.shape).astype(int)
        for i,sentence in enumerate(sentences):
            for j,word in enumerate(sentence):
                if word in self.word_to_id: sentences_inputs[i,j] = self.word_to_id[word]
                else: sentences_inputs[i,j] = self.word_to_id[UNK]
        for i,sentence_tags in enumerate(tags):
            for j,tag in enumerate(sentence_tags):
                if tag in self.tag_to_id:tags_inputs[i,j] = self.tag_to_id[tag]
                else: tags_inputs[i,j] = 0
        if mode=='test': return sentences_inputs,tags_inputs,None
        for i,sentence_labels in enumerate(labels):
            for j,label in enumerate(sentence_labels):
                labels_inputs[i,j] = self.label_to_id[label]
        return sentences_inputs,tags_inputs,labels_inputs

    def build_vocab(self,sentences,tags,labels):
        # build word-id dictionary
        counts = {}
        for sentence in sentences:
            for word in sentence:
                if word not in counts:counts[word]=1
                else: counts[word]+=1
        sorted_counts = sorted(counts.items(),key=lambda x:(-x[1],x[0]))
        word_to_id = {PAD:0}
        idx = 1
        for word,count in sorted_counts:
            if word not in word_to_id:
                word_to_id[word] = idx
                idx += 1
        word_to_id[UNK] = idx
        id_to_word = {v:k for k,v in word_to_id.items()}
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word

        # build label-id dictionary
        label_to_id={PAD:0}
        idx=1
        for sentence_label in labels:
            for label in sentence_label:
                if label not in label_to_id:
                    label_to_id[label] = idx
                    idx+=1
        id_to_label = {v:k for k,v in label_to_id.items()}
        self.label_to_id = label_to_id
        self.id_to_label = id_to_label

        # build tag-id dictionary
        tag_to_id = {NULL:0}
        idx=1
        for sentence_tag in tags:
            for tag in sentence_tag:
                if tag not in tag_to_id:
                    tag_to_id[tag] = idx
                    idx += 1
        id_to_tag = {v:k for k,v in tag_to_id.items()}
        self.tag_to_id = tag_to_id
        self.id_to_tag = id_to_tag

    def build_sentences(self,mode):
        sentences = []
        labels = []
        tags = []
        if mode=="test":
            with open(self.test_path,encoding='utf-8') as f:
                sentence=[]
                for line in f:
                    line = line.strip()
                    if not line:
                        sentences.append(sentence)
                        tags.append([pair[1] for pair in nltk.pos_tag(sentence)])
                        sentence = []
                        continue
                    sentence.append(line.lower())
        else:
            if mode=="train": file = open(self.train_label_path,encoding='utf-8')
            else: file = open(self.dev_label_path,encoding='utf-8')
            sentence=[]
            label=[]
            for line in file:
                line = line.split()
                if not line:
                    sentences.append(sentence)
                    tags.append([pair[1] for pair in nltk.pos_tag(sentence)])
                    labels.append(label)
                    sentence=[]
                    label=[]
                    continue
                word,wordlabel = line
                sentence.append(word.lower())
                label.append(wordlabel)
            file.close()
        return sentences,tags,labels

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile, 'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model

def build_worddict_glove_cache(gloveFile,worddict):
    print("Build loader's worddict embedding from glove")
    glove = loadGloveModel(gloveFile)
    cache = open('./data/worddict_glove.txt','w',encoding='utf-8')
    count = 0
    word=[]
    for k,v in worddict.items():
        if v in glove:
            for x in glove[v]:
                cache.write(str(x)+" ")
            cache.write('\n')
        else:
            count+=1
            word.append(v)
            y = np.random.rand(1,glove['hello'].shape[0])
            for x in y[0]:
                cache.write(str(x) + " ")
            cache.write('\n')
    cache.close()
    print("There are ",count," random initial embedding")
    print(word[:100])


if __name__ == '__main__':
    print("Load Data from file")
    loader = DataLoader()
    loader.load_data("train")
    train_x, train_pos, train_y = loader.load_data('train')
    dev_x, dev_pos, dev_y = loader.load_data('dev')
    test_x, test_pos, _ = loader.load_data('test')
    print('Train Shape: ',train_x.shape)
    print('Dev Shape: ', dev_x.shape)
    print('Test Shape: ', test_x.shape)
    if not os.path.exists('./data/worddict_glove.txt'):build_worddict_glove_cache("./data/glove.twitter.27B/glove.twitter.27B.50d.txt",loader.id_to_word)
