import numpy as np
import csv
import itertools
import string
table = str.maketrans({key: None for key in string.punctuation})

class Loader:
    def __init__(self, n_gram):
        self.word_path = './data/{}_x.csv'
        self.label_path = './data/{}_y.csv'
        self.tag_vocab = set(['*', '<STOP>'])
        self.rare_word=set()
        self.suffix_set=[]
        self.suffix_train=set()
        self.commom_word=set()
        self.train_word=[]
        self.ngram = n_gram

    def load_data(self, mode):
        sentences, labels = self.build_sentences(mode)
        return sentences, labels

    def build_sentences(self, mode):
        ## Build function for chunking span recongnization
        ones = [ "one","two","three","four", "five", "six","seven","eight","nine","ten","eleven","twelve", "thirteen", "fourteen", "fifteen","sixteen","seventeen", "eighteen","nineteen","twenty","thirty","forty", "fifty","sixty","seventy","eighty","ninety"]
        def is_number(n):
            try:
                float(n.translate(table)) 
            except ValueError:
                return False
            return True
        def is_alpha_number(s):
            output = (any(i.isdigit() for i in s)) & (any(i.islower() for i in s))
            if (any(i.isdigit() for i in s)) & (s[-1]=='s'):
                output=False
            return output
        def is_hyphen_alpha(s):
            output = ( ('-' in s)) & ( s.translate(table).isalpha()) & (s!='-DOCSTART-')
            if output:
                if any(i.isupper() for i in s.split('-')[-1]):
                    output = False
            return output
        def is_hyphen_Alpha(s):
            output = ( ('-' in s)) & ( s.translate(table).isalpha()) & (s!='-DOCSTART-')
            output2 =False
            if output:
                a=[]
                for word in s.split('-'):
                    a.append(any(i.isupper() for i in word))
                output2 =all(a)
            final_output = all([output,output2])
            return final_output
        def is_Upper_s(s):
            output = ( (s[:-1].isupper())) & ( s[-1:]=='s') & (len(s)>=3)
            return output
        def is_number_s(s):
            output=  (any(i.isdigit() for i in s)) & (s[-1]=='s')
            return output

        ngram = self.ngram
        sentences = []
        labels = []
        if mode == 'dev':
            with open(self.word_path.format(mode)) as f_input, open(self.label_path.format(mode)) as f_label:
                next(f_input)
                next(f_label)
                sentence = ['*'] * (ngram - 1)
                tags = ['*'] * (ngram - 1)
                for input_line, label_line in zip(csv.reader(f_input), csv.reader(f_label)):
                    tag = label_line[1]
                    word = input_line[1]
                    tags.append(tag)
                    sentence.append(word)
                    ## Collect tags
                    self.tag_vocab.add(tag) 
                    if (word == '.') or (word == '?') or (word == '!'):
                        sentences.append(sentence + ['<STOP>'])
                        labels.append(tags + ['<STOP>'])
                        sentence = ['*'] * (ngram - 1)
                        tags = ['*'] * (ngram - 1)

        if mode == 'test':
            with open(self.word_path.format(mode)) as f:
                next(f)
                sentence = ['*'] * (ngram - 1)
                for input_line in csv.reader(f):
                    word = input_line[1]
                    sentence.append(word)
                    if (word == '.') or (word == '?') or (word == '!'):
                        sentences.append(sentence + ['<STOP>'])
                        sentence = ['*'] * (ngram - 1)

        if mode == 'train':
            with open(self.word_path.format(mode)) as f_input, open(self.label_path.format(mode)) as f_label:
                next(f_input)
                next(f_label)
                sentence = ['*'] * (ngram - 1)
                tags = ['*'] * (ngram - 1)
                for input_line, label_line in zip(csv.reader(f_input), csv.reader(f_label)):
                    word = input_line[1]
                    tag = label_line[1]
                    sentence.append(word)
                    tags.append(tag)
                    self.tag_vocab.add(tag) # build vocab for tags
                    if (word == '.') or (word == '?') or (word == '!'):
                        sentences.append(sentence + ['<STOP>'])
                        labels.append(tags + ['<STOP>'])
                        sentence = ['*'] * (ngram - 1)
                        tags = ['*'] * (ngram - 1)
            ## Buid rare/unknown words training data
            ### Count the words
            counts = {}
            for sentence in sentences:
                for word in sentence:
                    if word == '*' or word == '<STOP>':
                        continue
                    if word not in counts:
                        counts[word] = 1
                    else:
                        counts[word] += 1
            threshold=2
            for word, count in counts.items():
                if count<=threshold:
                    self.rare_word.add(word)
                if count>threshold:
                    self.commom_word.add(word)

            X_all_gram_train = np.array(list(itertools.chain(*sentences)))
            train_size = len(X_all_gram_train)
            Y_all_gram_train = np.array(list(itertools.chain(*labels)))
            Y_label = sorted(set(Y_all_gram_train))
            label_size = len(Y_label)
            #print (X_all_gram_train.shape,Y_all_gram_train)
            # Training on the suffix unknown/rare word handling
            def suffix_n(n,threshold=5):
                suffix = {}
                output = {}
                for i in range (train_size):
                    word = X_all_gram_train[i]
                    if word not in self.commom_word:
                        if (is_number(word)) or (word.isupper()) or (is_alpha_number(word)) or (is_hyphen_alpha(word)) or (is_Upper_s(word)): continue 
                        if not (len(X_all_gram_train[i][-n:])==n):continue 
                        if X_all_gram_train[i][-n:] in suffix:
                            suffix[word[-n:]][Y_label.index(Y_all_gram_train[i])]+=1
                        else:
                            suffix[word[-n:]]=np.zeros(label_size)
                            suffix[word[-n:]][Y_label.index(Y_all_gram_train[i])]+=1
                for key,value in suffix.items():
                    size = float(sum(value))
                    mle = (max(value))/size
                    if size>=threshold and mle>0.3:
                        output[key]= mle
                return output
            #self.suffix_set.append(suffix_n(1))
            self.suffix_set.append(suffix_n(2))
            self.suffix_set.append(suffix_n(3))
            self.suffix_set.append(suffix_n(4))
            #self.suffix_set.append(suffix_n(5))
            #self.suffix_set.append(suffix_n(6))
            # print(self.suffix_set[0])
            # print(self.suffix_set[1]['ing'])
            # print(self.suffix_set[2]['ning'])


        ## Rare and unknown words handling
        new_sentences = sentences[:]
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                word = sentences[i][j]
                if word == '*' or word == '<STOP>':
                    continue
                if word in self.commom_word: continue
                if (is_number(word)) or (word.lower() in ones) : 
                    new_sentences[i][j]='*number*'
                    continue
                if word.isupper() and (word!='-DOCSTART-') and (word!='<STOP>') and (len(word) >=2) and ('-' not in word): 
                    new_sentences[i][j] = '*upper*'
                    continue
                if is_alpha_number(word): 
                    new_sentences[i][j]='*alpha_number*'
                    continue 
                if is_hyphen_alpha(word): 
                    new_sentences[i][j]='*hyphen_alpha*' 
                    continue
                if is_Upper_s(word): 
                    new_sentences[i][j]='*upper_s*' 
                    continue
                if is_number_s(word):
                    new_sentences[i][j]='*number_s*' 

                if is_hyphen_Alpha(word):
                    new_sentences[i][j]='*hyphen_Alpha*' 
                    continue
                if (word[0].isupper() and word[1:].islower() and (not any(i.isdigit() for i in word))) and (not '-' in word) and (word.lower() not in ones):
                    new_sentences[i][j]='*Alpha*'
                    continue 
                n_suffix=1
                mle = 0
                count=0
                for suffix_dic in self.suffix_set :
                    n_suffix+=1
                    if word[-n_suffix:] in suffix_dic:
                        count+=1
                        if suffix_dic[word[-n_suffix:]]>mle: 
                            mle = suffix_dic[word[-n_suffix:]]
                            suffix=word[-n_suffix:]
                if mode=='train':
                    if count>0:
                        new_sentences[i][j]='*' + suffix + '*'
                        self.suffix_train.add('*' + suffix + '*')
                    else:new_sentences[i][j]= '*unknown*'
                if mode=='test' or mode=='dev':
                    flag=0
                    if (count>0):
                        if ( ('*'+suffix + '*') in self.suffix_train ):
                            new_sentences[i][j]='*'+suffix + '*'
                            flag=1
                    if not flag:
                        new_sentences[i][j]= '*unknown*'
        # print (new_sentences[0])
        return new_sentences, labels


if __name__ == '__main__':
    loader = Loader(n_gram=2)
    x_train, y_train = loader.load_data('train')
    x_dev, y_dev = loader.load_data('dev')
    x_test, _ = loader.load_data('test')

