import dynet_config
import sys, os
import _dynet as dy
dyparams = dy.DynetParams()
dyparams.set_mem(40000)
dyparams.set_autobatch(True)
dyparams.init()
import numpy as np
import pandas as pd
import sys
import itertools
import string
from collections import Counter
from nltk.corpus import wordnet as wn
from collections import defaultdict
import nltk
import os
from scipy.stats import spearmanr
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
from numpy import random
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')
### Stop words and punction library     
from nltk.corpus import stopwords
stop = stopwords.words("english")
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

## Function to read conll file one sentence by one sentence
def read_conll(fh):
    root = (0,'*root*',-1,'rroot')
    tokens = [root]
    for line in fh:
        line = (line).decode('utf-8').lower()
        tok = line.strip().split()
        if not tok:
            if len(tokens)>1: yield tokens
            tokens = [root]
        else:
            tokens.append((int(tok[0]),tok[1],int(tok[-4]),tok[-3]))
    if len(tokens) > 1:
        yield tokens
        
## Parse Conll file to word, context pair        
def parse_conll(vocab,Conll):
    words = []
    contexts = []
    for i,sent in enumerate(read_conll(file(Conll))):
        for tok in sent[1:]:
            par = sent[tok[2]]
            m = (tok[1])
            if (m in stop): continue
            m=stemmer.stem(m)
            if (m not in vocab) or ((len(str(m)))<=1): continue
            rel = tok[3]
            if rel == 'adpmod': continue  # Not useful
            if rel == 'adpobj' and par[0] != 0:
                ppar = sent[par[2]]
                rel = "%s:%s" % (par[3],par[1])
                h = ppar[1]
            else:
                h = par[1]
            if h in stop: continue
            h = stemmer.stem(h)
            if (h not in vocab) or ((len(str(h)))<=1): continue
            words.append(h)
            words.append(m)
            contexts.append("_".join((rel,m)))
            contexts.append("I_".join((rel,h)))
    return np.array(words),np.array(contexts)

## Function to subsample data, make max count of every word as the threshold value by randomly sampling
def subsample(wc_window,filter_size=2000):
    random.shuffle(wc_window)
    wc_window_downsample = {}
    wc_window_d = []
    for word,context in wc_window:
        if word not in wc_window_downsample:
            wc_window_downsample[word]=1
        else:
            wc_window_downsample[word]+=1
        if wc_window_downsample[word]<=filter_size:
            wc_window_d.append(np.array([word,context]))
    (wc_window_d) = np.array(wc_window_d)
    return wc_window_d
        
def load_train_data(data_path='./data/training/training-data.1m',filter_number=10,mode='bow',window=2,Conll='./data/training/training-data.1m.conll',subsample_n=2000):
    ## Load the data and apply preprocessing: lowercase, remove punction, stop words, stem
    dataframe = pd.read_csv(data_path,sep='\t',header=None,encoding="utf8")
    dataframe[0] = dataframe[0].apply(lambda x: [stemmer.stem(word) for word in word_tokenize(x.lower().translate(remove_punctuation_map).encode("ascii", errors="ignore").decode()) if (word not in stop and word.isalpha()) ])
    all_words_train = np.array(list(itertools.chain(*dataframe[0].values)))
    unique_word_train = np.unique((all_words_train),return_counts=True)
    ## Filter the low frequency words
    filter_index = (np.where(unique_word_train[1]>=filter_number))[0]
    unique_word_train_filtered=(unique_word_train[0][filter_index],unique_word_train[1][filter_index])
    vocab = set(unique_word_train_filtered[0])
    vocab_sort = sorted(vocab)
    ## Dictionary for word and index
    vob2int = {vocab_sort[i]:i  for i in range(len(vocab_sort))}
    preprocessed_data = dataframe[0].apply(lambda x:' '.join([word for word in x if word in vocab])).values
    wc_int = []
    if mode=='bow':
    	## Generate pair of word and context given window size
        positions = [(x,"l%s_" % x) for x in xrange(-window, +window+1) if x != 0]
        for line in preprocessed_data:
            toks = ['<s>']
            toks.extend(line.strip().strip('"').split(' '))
            for i,tok in enumerate(toks):
                if tok not in vocab: continue
                for j,s in positions:
                    if i+j < 0: continue
                    if i+j >= len(toks): continue
                    c = toks[i+j]
                    if c not in vocab: continue
                    wc_int.append(np.array([vob2int[tok],vob2int[c]]))
        wc_int = np.array(wc_int)
        train_int = subsample(wc_int,filter_size=subsample_n)
        word_size = len(vocab)
        ## Get vocabulary size and get 3/4 power of the frequency for the future negative sampling
        vocab_count = Counter(train_int[:,0])
        vocab_count_array = np.array([float(vocab_count[word]) for word in range(len(vocab_sort))])
        context_fre=(vocab_count_array**(3.0/4))/np.sum(vocab_count_array**(3.0/4))
        context_size = len(vocab)
    elif mode=='dep':
    	## Generate dependency based word, context pair
        words,contexts = parse_conll(vocab,Conll)
        u_words = np.unique(words,return_counts=True )
        u_context = np.unique(contexts,return_counts=True) 
        ## Filter low frequency words and contexts
        words_filter = set(u_words[0][np.where(u_words[1]>=filter_number)])
        context_filter = set(u_context[0][np.where(u_context[1]>=filter_number)])
        vob2int = {vocab_sort[i]:i  for i in range(len(vocab_sort))}
        pair_filter = np.array([np.array([words[i],contexts[i]]) for i in range(words.shape[0]) if (words[i] in (words_filter)) and (contexts[i] in context_filter)])
        vocab = set(pair_filter[:,0])
        word_sorted = sorted(vocab)
        context_sorted = sorted(set(pair_filter[:,1]))
        word2int = {word_sorted[i]:i  for i in range(len(word_sorted))}
        context2int = {context_sorted[i]:i  for i in range(len(context_sorted))}
        ## Subsampling
        pair_sub = subsample(pair_filter,filter_size=subsample_n)
        train_int = np.array([np.array([word2int[x],context2int[y]]) for x,y in pair_filter])
        word_size = len(word_sorted)
        context_size = len(context_sorted)
        vocab_count = Counter(train_int[:,0])
        context_counts = Counter(train_int[:,1])
        context_count_array = np.array([ context_counts[context] for context in range(context_size)])
        context_fre=(context_count_array**(3.0/4))/np.sum(context_count_array**(3.0/4))
        
    else:
        raise Exception("mode should be set to 'bow' or 'dep'. 'bow' means linear bag of word method within the defined window size. 'dep' means depdendency-based context method.")
    return train_int,vocab,vocab_count ,context_fre,word_size, context_size,window,subsample_n



class Word2VecModel:
    def __init__(self,word_size,context_fre, context_size,vocab,window=2,subsample_n=2000,mode='bow',embed_size=200, batch_size=128,num_sampled=5, epoch=6):
        self.embed_size = embed_size
        self.mode = mode
        self.window = window
        self.vocab = vocab
        self.word_size = word_size
        self.subsample_n = subsample_n
        self.context_size = context_size
        self.num_sampled = num_sampled
        self.epoch = epoch
        self.context_fre = context_fre
        self.batch_size=batch_size
        self.pc = dy.ParameterCollection()
        self.optimizer = dy.AdamTrainer(self.pc)
        self.word_embeddings = self.pc.add_lookup_parameters((self.word_size, self.embed_size), name="word-embeddings")
        self.context_embeddings = self.pc.add_lookup_parameters((self.context_size, self.embed_size), name="context-embeddings")
        dy.renew_cg()
        print ([(param.name(), param.shape()) for param in self.pc.lookup_parameters_list() + self.pc.parameters_list()])
    def get_score(self,word,context):
    	## Get the loss given word, context pair and perform negative sampling
        objective = dy.logistic(((dy.transpose(self.context_embeddings[context]))*self.word_embeddings[word]))
        negative_sample = np.random.choice(self.context_size, self.num_sampled, replace=False, p=self.context_fre)
        for context_prime in negative_sample:
            objective *= dy.logistic(-((dy.transpose(self.context_embeddings[context_prime]))*self.word_embeddings[word]))
        loss = -dy.log(objective)
        return loss
    def epoch_train(self,examples):
        count=0
        dy.renew_cg()
        current_losses = [ ]
        loss_list = []
        for word,context in (examples):
            loss = self.get_score(word,context)
            current_losses.append(loss)
            loss_list.append(loss.value())
            if len(current_losses) >= self.batch_size:
                mean_loss = dy.esum(current_losses) / float(len(current_losses))
                mean_loss.forward()
                mean_loss.backward()
                self.optimizer.update()
                current_losses = [ ]
                dy.renew_cg()
            count+=1
            ## Print out the average loss in every 1M example
            if count%1000000==1000:
                print (count,np.mean(np.array(loss_list)))
                loss_list = []
        if current_losses:
            mean_loss = dy.esum(current_losses) / float(len(current_losses))
            mean_loss.forward()
            mean_loss.backward()
            self.optimizer.update()

    def train(self,data):
        for i in range(self.epoch):
            self.epoch_train(data)
            self.pc.save(self.mode + '_d' + '_window'+str(self.window) +'_'+str(self.embed_size)+'_n' +str(self.num_sampled )+'_epoch' +str(1+i)+'_down'+str()+'batch'+str(self.batch_size) +'_larger.model')
    def infer(self,word):
    	## Infer the most related words in vocab for unknwon words, first from the synset then based on the characters similarity 
        syn = list(itertools.chain(*[x.lemma_names() for x in wn.synsets(word)]))
        word_list = [stemmer.stem(x) for x in syn if (stemmer.stem(x) in self.vocab)]
        if not len(word_list): 
            word_list = [i for i in self.vocab if ((i in word) & (len(str(i))>=4) or ((stemmer.stem(word) in i) & len(stemmer.stem(word))>=4)) ]
        return word_list
        ## If output_all True, then the embedding wwill be all vocabulary, if False, only words in dev and test will be outputed.
    def save_embeddings(self,word_count,dev_train= './data/similarity/dev_x.csv',test_train='./data/similarity/test_x.csv',output_all=False):
        filename = 'embed_' + self.mode + '_d' + str(self.embed_size)+'_n' +str(self.num_sampled )+'_epoch' +str(self.epoch)+'_down'+str(self.subsample_n)+'_batch'+str(self.batch_size) +'_larger.csv'
        E_bed = {}
        ## Build dictionary for word and embedding
        for i,word in enumerate(sorted(self.vocab)):
            E_bed[word] = self.word_embeddings[i].npvalue()
        ## Include the embbeding for dev and test words
        test= (pd.read_csv(test_train ,sep=',',encoding="utf8",index_col = 0).values).flatten()
        dev= (pd.read_csv( dev_train,sep=',',encoding="utf8",index_col = 0).values).flatten()
        total_word = np.append(test,dev)
        if output_all: total_word = np.append(total_word,np.array(vocab))
       	total_word = set(total_word)
        with open (filename,'w') as OpenEmbedding:
            for i,words in enumerate(total_word):
                stem = stemmer.stem(words.lower())
                if stem in self.vocab:
                    E_bed[words] = np.array((E_bed[stem]))
                else:
                    word_list = self.infer(words)
                    if len(word_list): rare_e = np.mean((np.array([E_bed[word] for word in word_list])),axis=0)
                    else:
                        rare_e = np.mean((np.array([E_bed[word] for word in [word for word in self.vocab if int(word_count[word])<=35]])),axis=0)
                    E_bed[words] = rare_e
                E_str = ' '.join([str(e) for e in (E_bed[words])])
                OpenEmbedding.writelines('%s\t%s\n' % (words,E_str)) 
        return E_bed
    def evaluate(self,word_count,target='./data/similarity/dev_x.csv',label= './data/similarity/dev_y.csv',filename = 'prediction_dev.csv'):
        dev_dataframe = pd.read_csv(target,sep=',',encoding="utf8",index_col = 0)
        if label: actual = pd.read_csv(label,sep=',',encoding="utf8",index_col = 0).values.flatten()
        prediction = open (filename,'w')
        prediction.writelines('id,similarity\n')
        count = 0
        similarity = []
        Embedding = self.save_embeddings(word_count)                                 
        for w1, w2 in zip(dev_dataframe.word1, dev_dataframe.word2):
            value = np.dot(Embedding[w1],Embedding[w2])
            similarity.append(value)
            prediction.writelines('%d,%.5f\n' % (count,value))
            count+=1
        if label:
            print("Correlation:", spearmanr(list(actual),(similarity)).correlation)

if __name__ == "__main__":
	## Generate training data, you need to give:
		# Data path, 
		# Filter_number to get rid of words of low frequence
		# Context mode, dep for dependency, bow for linear bag of words
		# Window size for bag of words context model
		# CONLL data path for dependency context mode
		# subsample threshold size that reduces the high frequency words number to threshold
	train_int,vocab,vocab_count,context_fre,word_size,context_size,window,subsample_n = load_train_data(data_path='./data/training/training-data.1m',filter_number=10,mode='bow',window=2,Conll='./data/training/training-data.1m.conll',subsample_n=2000)
	## Training model and output reuslts
		# Embed_size, embedding dimension size, default is 300
		# Batch_size, default is 300
		# num_sampled for negative sampling, default is 5
		# Epoch of training, default is 6
	model = Word2VecModel(word_size,context_fre, context_size,vocab,window,subsample_n=2000,mode='bow',embed_size=300, batch_size=300,num_sampled=5, epoch=6)
	## If you want to continue precious model, input here
	#mode.pc.populate('xxx.model')
	model.train(train_int)
	## Provide traget file with word1, word2 and this function calculates similarity
	## and output file given the filename. If label is given, accuracy will be calculated.
	## if label is None, it only generates similarity file (for test).
	model.evaluate(vocab_count,target='./data/similarity/dev_x.csv',label= './data/similarity/dev_y.csv',filename = 'prediction.csv')
	model.evaluate(vocab_count,target='./data/similarity/test_x.csv',label= None,filename = 'test_y.csv')