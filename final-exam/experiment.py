
# coding: utf-8


import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import nltk
import os
from collections import OrderedDict
import visdom
import sys
import tqdm
import re,sys
import copy
UNK = '<unk>'
PAD = '<pad>'
NULL = 'NULL'
START_TAG = "<START>"
STOP_TAG = "<STOP>"


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
        if mode == 'train':
            self.build_vocab(sentences,tags,labels)
        return sentences,labels

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
                    if len(sentence)==0:continue
                    sentences.append(sentence)
                    tags.append([pair[1] for pair in nltk.pos_tag(sentence)])
                    labels.append(label)
                    sentence=[]
                    label=[]
                    continue
                word,wordlabel = line
                sentence.append(word.lower())
                label.append(wordlabel[0])
            file.close()
        return sentences,tags,labels

def prepare_dataset(sentences,labels, word_to_id, char_to_id, tag_to_id, lower=True):
    def f(x): return x.lower() if lower else x
    data = []
    for i in range(len(sentences)):
        s = sentences[i]
        str_words = [w for w in s]
        words = [word_to_id[f(w) if f(w) in word_to_id else UNK]
                 for w in str_words]
        # Skip characters that are not in the training set
        chars = [[char_to_id[c if c in char_to_id else UNK] for c in w] for w in str_words]
        caps = [cap_feature(w) for w in str_words]
        if len(labels)!=0:tags = [tag_to_id[w] for w in labels[i]]
        else: tags=[]
        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'caps': caps,
            'tags': tags,
        })
    return data

def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3

# evaluate f1,prec,recall
def warning(msg):
    print(sys.stderr, "WARNING:", msg)

def convert_bio_to_spans(bio_sequence):
    spans = []  # (label, startindex, endindex)
    cur_start = None
    cur_label = None
    N = len(bio_sequence)
    for t in range(N+1):
        if ((cur_start is not None) and
                (t==N or re.search("^[BO]", bio_sequence[t]))):
            assert cur_label is not None
            spans.append((cur_label, cur_start, t))
            cur_start = None
            cur_label = None
        if t==N: continue
        assert bio_sequence[t] and bio_sequence[t][0] in ("B","I","O")
        if bio_sequence[t].startswith("B"):
            cur_start = t
            cur_label = re.sub("^B-?","", bio_sequence[t]).strip()
        if bio_sequence[t].startswith("I"):
            if cur_start is None:
                warning("BIO inconsistency: I without starting B. Rewriting to B.")
                newseq = bio_sequence[:]
                newseq[t] = "B" + newseq[t][1:]
                return convert_bio_to_spans(newseq)
            continuation_label = re.sub("^I-?","",bio_sequence[t])
            if continuation_label != cur_label:
                newseq = bio_sequence[:]
                newseq[t] = "B" + newseq[t][1:]
                warning("BIO inconsistency: %s but current label is '%s'. Rewriting to %s" % (bio_sequence[t], cur_label, newseq[t]))
                return convert_bio_to_spans(newseq)

    # should have exited for last span ending at end by now
    assert cur_start is None
    spancheck(spans)
    return spans

def spancheck(spanlist):
    s = set(spanlist)
    assert len(s)==len(spanlist), "spans are non-unique ... is this a bug in the eval script?"
    
def kill_labels(bio_seq):
    ret = []
    for x in bio_seq:
        if re.search("^[BI]", x):
            x = re.sub("^B.*","B", x)
            x = re.sub("^I.*","I", x)
        ret.append(x)
    return ret

def evaluate_taggings(goldseq_predseq_pairs, ignore_labels=False):
    """a list of (goldtags,predtags) pairs.  goldtags and predtags are both lists of strings, of the same length."""
    num_sent = 0
    num_tokens= 0
    num_goldspans = 0
    num_predspans = 0

    tp, fp, fn = 0,0,0

    for goldseq,predseq in goldseq_predseq_pairs:
        N = len(goldseq)
        assert N==len(predseq)
        num_sent += 1
        num_tokens += N

        if ignore_labels:
            goldseq = kill_labels(goldseq)
            predseq = kill_labels(predseq)

        goldspans = convert_bio_to_spans(goldseq)
        predspans = convert_bio_to_spans(predseq)

        num_goldspans += len(goldspans)
        num_predspans += len(predspans)

        goldspans_set = set(goldspans)
        predspans_set = set(predspans)

        # tp: number of spans that gold and pred have
        # fp: number of spans that pred had that gold didn't (incorrect predictions)
        # fn: number of spans that gold had that pred didn't (didn't recall)
        tp += len(goldspans_set & predspans_set)
        fp += len(predspans_set - goldspans_set)
        fn += len(goldspans_set - predspans_set)

    prec = tp/(tp+fp) if (tp+fp)>0 else 1
    rec =  tp/(tp+fn) if (tp+fn)>0 else 1
    f1 = 2*prec*rec / (prec + rec)
    return f1,prec,rec

def evaluating(model, datas):
    prediction = []
    y=[]
    for data in datas:
        ground_truth_id = data['tags']
        words = data['str_words']
        chars2 = data['chars']
        caps = data['caps']
        
        if parameters['char_mode'] == 'LSTM':
            chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
            d = {}
            for i, ci in enumerate(chars2):
                for j, cj in enumerate(chars2_sorted):
                    if ci == cj and not j in d and not i in d.values():
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in chars2_sorted]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
            for i, c in enumerate(chars2_sorted):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))
        if parameters['char_mode'] == 'CNN':
            d = {}
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))
        dwords = Variable(torch.LongTensor(data['words']))
        dcaps = Variable(torch.LongTensor(caps))
        val, out = model(dwords, chars2_mask, dcaps, chars2_length, d)
        predicted_id = out
        prediction.append(out)
        y.append(ground_truth_id)
    return prediction,y

def calculatef1(y,p,id_to_class):
    #     true_pos = 0
    #     false_pos = 0
    #     false_neg = 0
    #     for y_seq,p_seq in zip(y,p):
    #         for y_id,p_id in zip(y_seq,p_seq):
    #             y_ = id_to_class[y_id]
    #             p_ = id_to_class[p_id]
    #             if y_ != 'O':
    #                 if p_ == y_:
    #                     true_pos += 1
    #                 elif p_ == 'O':
    #                     false_neg += 1
    #             elif p_ != y_:
    #                 false_pos += 1
    #     prec = true_pos / (true_pos + false_pos) if true_pos + false_pos != 0 else 0
    #     recall = true_pos / (true_pos + false_neg) if true_pos + false_neg != 0 else 0
    #     f1 = 2 * prec * recall / (prec + recall) if prec + recall != 0 else 0
    #     return f1, prec, recall
    y_label=[]
    p_label=[]
    for y_seq,p_seq in zip(y,p):
        y_=[]
        p_=[]
        for y_id,p_id in zip(y_seq,p_seq):
            y_.append(id_to_class[y_id])
            p_.append(id_to_class[p_id])
        y_label.append(y_)
        p_label.append(p_)
    return evaluate_taggings(zip(y_label,p_label), ignore_labels=False)

# Model

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform(input_embedding, -bias, bias)

def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


def init_lstm(input_lstm):
    """
    Initialize lstm
    """
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
    if input_lstm.bidirectional:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform(weight, -bias, bias)
            weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform(weight, -bias, bias)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                weight = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                weight = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, char_lstm_dim=25,
                 char_to_ix=None, pre_word_embeds=None, char_embedding_dim=None, use_gpu=False,
                 n_cap=None, cap_embedding_dim=None, use_crf=True, char_mode='CNN'):
        super(BiLSTM_CRF, self).__init__()
        self.use_gpu = use_gpu
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.n_cap = n_cap
        self.cap_embedding_dim = cap_embedding_dim
        self.use_crf = use_crf
        self.tagset_size = len(tag_to_ix)
        self.out_channels = char_lstm_dim
        self.char_mode = char_mode

        print('char_mode: %s, out_channels: %d, hidden_dim: %d, ' % (char_mode, char_lstm_dim, hidden_dim))

        if self.n_cap and self.cap_embedding_dim:
            self.cap_embeds = nn.Embedding(self.n_cap, self.cap_embedding_dim)
            init_embedding(self.cap_embeds.weight)

        if char_embedding_dim is not None:
            self.char_lstm_dim = char_lstm_dim
            self.char_embeds = nn.Embedding(len(char_to_ix), char_embedding_dim)
            init_embedding(self.char_embeds.weight)
            if self.char_mode == 'LSTM':
                self.char_lstm = nn.LSTM(char_embedding_dim, char_lstm_dim, num_layers=1, bidirectional=True)
                init_lstm(self.char_lstm)
            if self.char_mode == 'CNN':
                self.char_cnn3 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(3, char_embedding_dim), padding=(2,0))

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        if pre_word_embeds is not None:
            self.pre_word_embeds = True
            self.word_embeds.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))
        else:
            self.pre_word_embeds = False

        self.dropout = nn.Dropout(0.5)
        if self.n_cap and self.cap_embedding_dim:
            if self.char_mode == 'LSTM':
                self.lstm = nn.LSTM(embedding_dim+char_lstm_dim*2+cap_embedding_dim, hidden_dim, bidirectional=True)
            if self.char_mode == 'CNN':
                self.lstm = nn.LSTM(embedding_dim+self.out_channels+cap_embedding_dim, hidden_dim, bidirectional=True)
        else:
            if self.char_mode == 'LSTM':
                self.lstm = nn.LSTM(embedding_dim+char_lstm_dim*2, hidden_dim, bidirectional=True)
            if self.char_mode == 'CNN':
                self.lstm = nn.LSTM(embedding_dim+self.out_channels, hidden_dim, bidirectional=True)
        init_lstm(self.lstm)
        self.hw_trans = nn.Linear(self.out_channels, self.out_channels)
        self.hw_gate = nn.Linear(self.out_channels, self.out_channels)
        self.h2_h1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.tanh = nn.Tanh()
        self.hidden2tag = nn.Linear(hidden_dim*2, self.tagset_size)
        init_linear(self.h2_h1)
        init_linear(self.hidden2tag)
        init_linear(self.hw_gate)
        init_linear(self.hw_trans)

        if self.use_crf:
            self.transitions = nn.Parameter(
                torch.zeros(self.tagset_size, self.tagset_size))
            self.transitions.data[tag_to_ix[START_TAG], :] = -10000
            self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

    def _score_sentence(self, feats, tags):
        # tags is ground_truth, a list of ints, length is len(sentence)
        # feats is a 2D tensor, len(sentence) * tagset_size
        r = torch.LongTensor(range(feats.size()[0]))
        if self.use_gpu:
            r = r.cuda()
            pad_start_tags = torch.cat([torch.cuda.LongTensor([self.tag_to_ix[START_TAG]]), tags])
            pad_stop_tags = torch.cat([tags, torch.cuda.LongTensor([self.tag_to_ix[STOP_TAG]])])
        else:
            pad_start_tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
            pad_stop_tags = torch.cat([tags, torch.LongTensor([self.tag_to_ix[STOP_TAG]])])

        score = torch.sum(self.transitions[pad_stop_tags, pad_start_tags]) + torch.sum(feats[r, tags])

        return score

    def _get_lstm_features(self, sentence, chars2, caps, chars2_length, d):

        if self.char_mode == 'LSTM':
            # self.char_lstm_hidden = self.init_lstm_hidden(dim=self.char_lstm_dim, bidirection=True, batchsize=chars2.size(0))
            chars_embeds = self.char_embeds(chars2).transpose(0, 1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, chars2_length)
            lstm_out, _ = self.char_lstm(packed)
            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
            outputs = outputs.transpose(0, 1)
            chars_embeds_temp = Variable(torch.FloatTensor(torch.zeros((outputs.size(0), outputs.size(2)))))
            if self.use_gpu:
                chars_embeds_temp = chars_embeds_temp.cuda()
            for i, index in enumerate(output_lengths):
                chars_embeds_temp[i] = torch.cat((outputs[i, index-1, :self.char_lstm_dim], outputs[i, 0, self.char_lstm_dim:]))
            chars_embeds = chars_embeds_temp.clone()
            for i in range(chars_embeds.size(0)):
                chars_embeds[d[i]] = chars_embeds_temp[i]

        if self.char_mode == 'CNN':
            chars_embeds = self.char_embeds(chars2).unsqueeze(1)
            chars_cnn_out3 = self.char_cnn3(chars_embeds)
            chars_embeds = nn.functional.max_pool2d(chars_cnn_out3,
                                                 kernel_size=(chars_cnn_out3.size(2), 1)).view(chars_cnn_out3.size(0), self.out_channels)


        embeds = self.word_embeds(sentence)
        if self.n_cap and self.cap_embedding_dim:
            cap_embedding = self.cap_embeds(caps)

        if self.n_cap and self.cap_embedding_dim:
            embeds = torch.cat((embeds, chars_embeds, cap_embedding), 1)
        else:
            embeds = torch.cat((embeds, chars_embeds), 1)

        embeds = embeds.unsqueeze(1)
        embeds = self.dropout(embeds)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim*2)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _forward_alg(self, feats):
        # calculate in log domain
        # feats is len(sentence) * tagset_size
        # initialize alpha with a Tensor with values all equal to -10000.
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        forward_var = autograd.Variable(init_alphas)
        if self.use_gpu:
            forward_var = forward_var.cuda()
        for feat in feats:
            emit_score = feat.view(-1, 1)
            tag_var = forward_var + self.transitions + emit_score
            max_tag_var, _ = torch.max(tag_var, dim=1)
            tag_var = tag_var - max_tag_var.view(-1, 1)
            forward_var = max_tag_var + torch.log(torch.sum(torch.exp(tag_var), dim=1)).view(1, -1) # ).view(1, -1)
        terminal_var = (forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]).view(1, -1)
        alpha = log_sum_exp(terminal_var)
        # Z(x)
        return alpha

    def viterbi_decode(self, feats):
        backpointers = []
        # analogous to forward
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        forward_var = Variable(init_vvars)
        if self.use_gpu:
            forward_var = forward_var.cuda()
        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            bptrs_t = bptrs_t.squeeze().data.cpu().numpy()
            next_tag_var = next_tag_var.data.cpu().numpy()
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            viterbivars_t = Variable(torch.FloatTensor(viterbivars_t))
            if self.use_gpu:
                viterbivars_t = viterbivars_t.cuda()
            forward_var = viterbivars_t + feat
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        terminal_var.data[self.tag_to_ix[STOP_TAG]] = -10000.
        terminal_var.data[self.tag_to_ix[START_TAG]] = -10000.
        best_tag_id = argmax(terminal_var.unsqueeze(0))
        path_score = terminal_var[best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags, chars2, caps, chars2_length, d):
        # sentence, tags is a list of ints
        # features is a 2D tensor, len(sentence) * self.tagset_size
        feats = self._get_lstm_features(sentence, chars2, caps, chars2_length, d)

        if self.use_crf:
            forward_score = self._forward_alg(feats)
            gold_score = self._score_sentence(feats, tags)
            return forward_score - gold_score
        else:
            tags = Variable(tags)
            scores = nn.functional.cross_entropy(feats, tags)
            return scores


    def forward(self, sentence, chars, caps, chars2_length, d):
        feats = self._get_lstm_features(sentence, chars, caps, chars2_length, d)
        # viterbi to get tag_seq
        if self.use_crf:
            score, tag_seq = self.viterbi_decode(feats)
        else:
            score, tag_seq = torch.max(feats, 1)
            tag_seq = list(tag_seq.cpu().data)

        return score, tag_seq

'''
    Load Data
'''


print("Load Data from file")
loader = Loader()
train,train_labels = loader.load_data('train')
dev,dev_labels = loader.load_data('dev')
test,test_labels = loader.load_data('test')
train_data = prepare_dataset(train,train_labels, loader.word_to_id, loader.char_to_id, loader.label_to_id)
dev_data = prepare_dataset(dev,dev_labels, loader.word_to_id, loader.char_to_id, loader.label_to_id)
test_data = prepare_dataset(test,test_labels, loader.word_to_id, loader.char_to_id, loader.label_to_id)
print("Load Data Done")
print("Train ",len(train))
print("Dev ",len(dev))
print("Test ",len(test))

'''
    Adjust Parameters Here
'''

print("Set Experiment Parameter")
parameters = OrderedDict()
parameters['char_lstm_dim'] = 100 # char hidden dim
parameters['word_dim'] = 100
parameters['word_lstm_dim'] = 100 # word hidden
parameters['pre_emb'] = 'data/glove.6B/glove.6B.100d.txt'
parameters['cap_dim'] = 25 # control orth can be None or integer,25
parameters['char_embed_dim'] = 100 # control character can be None or integer,10
parameters['crf'] = 1
parameters['dropout'] = 0.5
parameters['char_mode'] = 'LSTM'
parameters['learning_rate'] = 0.001
parameters['epoch'] = 30
print(parameters)


'''
    Load Pretrained Word Embedding
'''


all_word_embeds = {}
file = open(parameters['pre_emb'],encoding = 'utf-8')
for line in file:
    s = line.strip().split()
    if len(s) == parameters['word_dim'] + 1:
        all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])
file.close()
word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(loader.word_to_id), parameters['word_dim']))

for w in loader.word_to_id:
    if w in all_word_embeds:
        word_embeds[loader.word_to_id[w]] = all_word_embeds[w]
    elif w.lower() in all_word_embeds:
        word_embeds[loader.word_to_id[w]] = all_word_embeds[w.lower()]

print('Loaded %i pretrained embeddings.' % len(all_word_embeds))

'''
    Train Model
'''


print("Initial Model")
model = BiLSTM_CRF(vocab_size=len(loader.word_to_id),
                   tag_to_ix=loader.label_to_id,
                   embedding_dim=parameters['word_dim'],
                   hidden_dim=parameters['word_lstm_dim'],
                   char_lstm_dim = parameters['char_lstm_dim'],
                   char_to_ix = loader.char_to_id,
                   pre_word_embeds=word_embeds,
                   char_embedding_dim=parameters['char_embed_dim'],
                   use_crf=parameters['crf'],
                   char_mode=parameters['char_mode'],
                   n_cap=4,
                   cap_embedding_dim=parameters['cap_dim'])
model.train(True)
optimizer = torch.optim.SGD(model.parameters(), lr=parameters['learning_rate'], momentum=0.9)
best_model = None
best_f1 = 0
best_dev_prediction=None
best_dev_y = None


print("Begin Training")
for epoch in range(parameters['epoch']):
    for i, index in tqdm.tqdm(enumerate(np.random.permutation(len(train_data)))):
        data = train_data[index]
        model.zero_grad()

        sentence_in = data['words']
        sentence_in = Variable(torch.LongTensor(sentence_in))
        tags = data['tags']
        chars2 = data['chars']

        if parameters['char_mode'] == 'LSTM':
            chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
            d = {}
            for i, ci in enumerate(chars2):
                for j, cj in enumerate(chars2_sorted):
                    if ci == cj and not j in d and not i in d.values():
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in chars2_sorted]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
            for i, c in enumerate(chars2_sorted):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))
        
        if parameters['char_mode'] == 'CNN':
            d = {}
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        targets = torch.LongTensor(tags)
        caps = Variable(torch.LongTensor(data['caps']))
        
        neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets, chars2_mask, caps, chars2_length, d)
        neg_log_likelihood.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        optimizer.step()
# evaluate and save model
    prediction,y = evaluating(model, dev_data)
    f1, prec, recall = calculatef1(y,prediction,loader.id_to_label)
    if f1>best_f1:
        best_f1 = f1
        best_model = copy.deepcopy(model)
        best_dev_prediction = copy.deepcopy(prediction)
        best_dev_y = copy.deepcopy(y)
        print("Better Model",f1, prec, recall)


print("DEV Error Analysis")
def error_analysis(y,prediction):
    coordinates=[]
    for i in range(len(y)):
        for j in range(len(y[i])):
            if y[i][j]!=prediction[i][j]:coordinates.append([i,j])
    return coordinates
coordinates = error_analysis(best_dev_y,best_dev_prediction)
for i in coordinates:
    print(dev[i[0]])
    print(dev[i[0]][i[1]],loader.id_to_label[best_dev_y[i[0]][i[1]]],loader.id_to_label[best_dev_prediction[i[0]][i[1]]])

# write predictions

print("Write Test Predictions")
model = copy.deepcopy(best_model)
test_predict = []
for data in test_data:
    ground_truth_id = data['tags']
    words = data['str_words']
    chars2 = data['chars']
    caps = data['caps']
    if parameters['char_mode'] == 'LSTM':
        chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
        d = {}
        for i, ci in enumerate(chars2):
            for j, cj in enumerate(chars2_sorted):
                if ci == cj and not j in d and not i in d.values():
                    d[j] = i
                    continue
        chars2_length = [len(c) for c in chars2_sorted]
        char_maxl = max(chars2_length)
        chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
        for i, c in enumerate(chars2_sorted):
            chars2_mask[i, :chars2_length[i]] = c
        chars2_mask = Variable(torch.LongTensor(chars2_mask))
    if parameters['char_mode'] == 'CNN':
        d = {}
        chars2_length = [len(c) for c in chars2]
        char_maxl = max(chars2_length)
        chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
        for i, c in enumerate(chars2):
            chars2_mask[i, :chars2_length[i]] = c
        chars2_mask = Variable(torch.LongTensor(chars2_mask))
    dwords = Variable(torch.LongTensor(data['words']))
    dcaps = Variable(torch.LongTensor(caps))
    val, out = model(dwords, chars2_mask, dcaps, chars2_length, d)
    test_predict.append(out)

file = open("test_predictions.out",'w',encoding='utf-8')
for i in test_predict:
    for j in i:
        file.write(loader.id_to_label[j])
        file.write('\n')
    file.write('\n')
file.close()

