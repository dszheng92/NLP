from DataLoader import DataLoader,Loader
from model import BiLSTM_CRF
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import tqdm

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

def build_worddict_embedding(gloveFile,worddict):
    print("Build loader's worddict embedding from glove")
    glove = loadGloveModel(gloveFile)
    embedding = []
    count = 0
    dim = np.sqrt(3.0 /glove['hello'].shape[0])
    for k,v in worddict.items():
        if v in glove:
            embedding.append(glove[v])
        else:
            count+=1
            y = np.random.uniform(-dim,dim,glove['hello'].shape[0])
            embedding.append(y)
    embedding = np.array(embedding)
    print(embedding.shape)
    print("There are ",count," random initial embedding")
    return embedding

def calculatef1(y,p,id_to_class):
    true_pos = 0
    false_pos = 0
    false_neg = 0
    for y_seq,p_seq in zip(y,p):
        for y_id,p_id in zip(y_seq,p_seq):
            y_ = id_to_class[y_id]
            p_ = id_to_class[p_id]
            if y_ != 'O':
                if p_ == y_:
                    true_pos += 1
                elif p_ == 'O':
                    false_neg += 1
            elif p_ != y_:
                false_pos += 1
    prec = true_pos / (true_pos + false_pos) if true_pos + false_pos != 0 else 0
    recall = true_pos / (true_pos + false_neg) if true_pos + false_neg != 0 else 0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall != 0 else 0
    return f1, prec, recall


if __name__ == '__main__':
    print("Load Data from file")
    loader = Loader()
    train_len, train_word, train_tag, train_char, train_orth, train_label = loader.load_data('train')
    print("Train: ",train_word.shape[0])
    dev_len,dev_word, dev_tag, dev_char, dev_orth, dev_label = loader.load_data('dev')
    print("Dev: ",dev_word.shape[0])
    test_len, test_word, test_tag, test_char, test_orth, test_label = loader.load_data('test')
    print("Test: ", test_word.shape[0])

    EMBEDDING_DIM = 200
    HIDDEN_DIM = 100

    model = BiLSTM_CRF(len(loader.word_to_id), loader.label_to_id, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    for epoch in range(50):
        print("Epoch",epoch)
        i=0
        for sentence, labels in tqdm.tqdm(zip(train_word,train_label)):
            model.zero_grad()
            sentence_in = torch.tensor(sentence[:train_len[i]], dtype=torch.long)
            targets = torch.tensor(labels[:train_len[i]], dtype=torch.long)
            loss = model.neg_log_likelihood(sentence_in, targets)
            loss.backward()
            optimizer.step()
            i+=1
            if(i%1000==0):print(loss)
        print()

    dev_predict=[]
    dev_y=[]
    i=0
    for sentence, labels in zip(dev_word, dev_label):
        dev_y.append(labels[:dev_len[i]])
        dev_predict.append(model(torch.tensor(sentence[:dev_len[i]], dtype=torch.long))[1])
        i+=1
    f1, prec, recall = calculatef1(dev_y,dev_predict,loader.id_to_label)
    print(f1, prec, recall)

    test_predict = []
    i=0
    for sentence in test_word:
        test_predict.append(model(torch.tensor(sentence[:test_len[i]], dtype=torch.long))[1])
        i+=1
    file = open("test.out",'w',encoding='utf-8')
    for i in test_predict:
        for j in i:
            file.write(loader.id_to_label[j])
            file.write('\n')
        file.write('\n')
    file.close()


