import numpy as np
import nltk
import os
import sys
UNK = '<unk>'
PAD = '<pad>'
NULL = 'NULL'
START_TAG = "<START>"
STOP_TAG = "<STOP>"

class DataLoader:
    # word, tag, character, orth, label
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
        self.char_to_id = None
        self.id_to_char = None
        self.orth_to_id = None
        self.id_to_orth = None
        self.max_len = 52

    def load_data(self,mode):
        sentences, tags, labels = self.build_sentences(mode)
        if mode == 'train':self.build_vocab(sentences,tags,labels)
        word_input,tag_input,char_input,orth_input,label_output = self.build_data(sentences, tags, labels, mode)
        sentence_len = []
        for sentence in sentences:sentence_len.append(len(sentence))
        sentence_len = np.array(sentence_len)
        return sentence_len,word_input,tag_input,char_input,orth_input,label_output

    def build_data(self,sentences, tags, labels, mode):
        word_input = np.zeros((len(sentences),self.max_len)).astype(int)
        tag_input = np.zeros(word_input.shape).astype(int)
        label_output = np.zeros(word_input.shape).astype(int)
        char_input = np.zeros((len(sentences), self.max_len, len(self.char_to_id))).astype(int)
        orth_input = np.zeros((len(sentences),self.max_len)).astype(int)
        for i,sentence in enumerate(sentences):
            for j,word in enumerate(sentence):
                if word in self.word_to_id:word_input[i,j] = self.word_to_id[word]
                else:word_input[i,j] = self.word_to_id[UNK]
                orthword = self.orthographic(word)
                if orthword in self.orth_to_id: orth_input[i,j] = self.orth_to_id[orthword]
                else: orth_input[i,j] = self.orth_to_id[UNK]
                for k,char in enumerate(word):
                    if char in self.char_to_id: char_input[i,j,self.char_to_id[char]]+=1
                    else:char_input[i,j,self.char_to_id[UNK]]+=1
        for i,sentence_tag in enumerate(tags):
            for j,tag in enumerate(sentence_tag):
                if tag in self.tag_to_id:tag_input[i,j] = self.tag_to_id[tag]
        if mode=='test':return word_input, tag_input, char_input, orth_input, None
        for i,sentence_label in enumerate(labels):
            for j,label in enumerate(sentence_label):
                label_output[i,j] = self.label_to_id[label]
        return word_input, tag_input, char_input, orth_input, label_output

    def build_vocab(self,sentences,tags,labels):
        # build word-id char-id orth-id dictionary
        counts = {}
        char_to_id = {PAD:0}
        orth_to_id = {PAD:0}
        charidx = 1
        orthidx = 1
        for sentence in sentences:
            for word in sentence:
                if word not in counts:counts[word]=1
                else: counts[word]+=1
                # orthographic
                orthword = self.orthographic(word)
                if orthword not in orth_to_id:
                    orth_to_id[orthword] = orthidx
                    orthidx += 1
                # character
                for char in word:
                    if char not in char_to_id:
                        char_to_id[char] = charidx
                        charidx+=1
        char_to_id[UNK] = charidx
        id_to_char = {v:k for k,v in char_to_id.items()}
        self.char_to_id = char_to_id
        self.id_to_char = id_to_char
        orth_to_id[UNK] = orthidx
        id_to_orth = {v:k for k,v in orth_to_id.items()}
        self.orth_to_id = orth_to_id
        self.id_to_orth = id_to_orth
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

    def orthographic(self, word):
        ortho_word = ''
        for char in word:
            if char.isupper():
                ortho_word += 'C'
            elif char.islower():
                ortho_word += 'c'
            elif char.isdigit():
                ortho_word += 'n'
            else:
                ortho_word += 'p'
        return ortho_word

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


class Loader:
    # word, tag, character, orth, label
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
        self.char_to_id = None
        self.id_to_char = None
        self.orth_to_id = None
        self.id_to_orth = None
        self.max_len = 52

    def load_data(self,mode):
        sentences, tags, labels = self.build_sentences(mode)
        if mode == 'train':self.build_vocab(sentences,tags,labels)
        word_input,tag_input,char_input,orth_input,label_output = self.build_data(sentences, tags, labels, mode)
        sentence_len = []
        for sentence in sentences:sentence_len.append(len(sentence))
        sentence_len = np.array(sentence_len)
        return sentence_len,word_input,tag_input,char_input,orth_input,label_output

    def build_data(self,sentences, tags, labels, mode):
        word_input = np.zeros((len(sentences),self.max_len)).astype(int)
        tag_input = np.zeros(word_input.shape).astype(int)
        label_output = np.zeros(word_input.shape).astype(int)
        char_input = np.zeros((len(sentences), self.max_len, len(self.char_to_id))).astype(int)
        orth_input = np.zeros((len(sentences),self.max_len)).astype(int)
        for i,sentence in enumerate(sentences):
            for j,word in enumerate(sentence):
                if word in self.word_to_id:word_input[i,j] = self.word_to_id[word]
                else:word_input[i,j] = self.word_to_id[UNK]
                orthword = self.orthographic(word)
                if orthword in self.orth_to_id: orth_input[i,j] = self.orth_to_id[orthword]
                else: orth_input[i,j] = self.orth_to_id[UNK]
                for k,char in enumerate(word):
                    if char in self.char_to_id: char_input[i,j,self.char_to_id[char]]+=1
                    else:char_input[i,j,self.char_to_id[UNK]]+=1
        for i,sentence_tag in enumerate(tags):
            for j,tag in enumerate(sentence_tag):
                if tag in self.tag_to_id:tag_input[i,j] = self.tag_to_id[tag]
        if mode=='test':return word_input, tag_input, char_input, orth_input, None
        for i,sentence_label in enumerate(labels):
            for j,label in enumerate(sentence_label):
                label_output[i,j] = self.label_to_id[label]
        return word_input, tag_input, char_input, orth_input, label_output

    def build_vocab(self,sentences,tags,labels):
        # build word-id char-id orth-id dictionary
        counts = {}
        char_to_id = {}
        orth_to_id = {}
        charidx = 0
        orthidx = 0
        for sentence in sentences:
            for word in sentence:
                if word not in counts:counts[word]=1
                else: counts[word]+=1
                # orthographic
                orthword = self.orthographic(word)
                if orthword not in orth_to_id:
                    orth_to_id[orthword] = orthidx
                    orthidx += 1
                # character
                for char in word:
                    if char not in char_to_id:
                        char_to_id[char] = charidx
                        charidx+=1
        char_to_id[UNK] = charidx
        id_to_char = {v:k for k,v in char_to_id.items()}
        self.char_to_id = char_to_id
        self.id_to_char = id_to_char
        orth_to_id[UNK] = orthidx
        id_to_orth = {v:k for k,v in orth_to_id.items()}
        self.orth_to_id = orth_to_id
        self.id_to_orth = id_to_orth
        sorted_counts = sorted(counts.items(),key=lambda x:(-x[1],x[0]))
        word_to_id = {}
        idx = 0
        for word,count in sorted_counts:
            if word not in word_to_id:
                word_to_id[word] = idx
                idx += 1
        word_to_id[UNK] = idx
        id_to_word = {v:k for k,v in word_to_id.items()}
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word

        # build label-id dictionary
        label_to_id={START_TAG:0}
        idx=1
        for sentence_label in labels:
            for label in sentence_label:
                if label not in label_to_id:
                    label_to_id[label] = idx
                    idx+=1
        label_to_id[STOP_TAG]=idx
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

    def orthographic(self, word):
        ortho_word = ''
        for char in word:
            if char.isupper():
                ortho_word += 'C'
            elif char.islower():
                ortho_word += 'c'
            elif char.isdigit():
                ortho_word += 'n'
            else:
                ortho_word += 'p'
        return ortho_word

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


if __name__ == '__main__':
    print("Load Data from file")
    loader = DataLoader()
    train_len, train_word, train_tag, train_char, train_orth, train_label = loader.load_data('train')
    print("Train: ",train_word.shape[0])
    dev_len,dev_word, dev_tag, dev_char, dev_orth, dev_label = loader.load_data('dev')
    print("Dev: ",dev_word.shape[0])
    test_len, test_word, test_tag, test_char, test_orth, test_label = loader.load_data('test')
    print("Test: ", test_word.shape[0])
    print(loader.label_to_id)
