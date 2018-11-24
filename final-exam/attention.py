import numpy as np
import json
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import string
translator = str.maketrans('', '', string.punctuation)
import dynet as dy
import random


"'adasd;'".translate(translator)

x_train = json.load(open('train.json'))

x_train[:1]

ins_exp = [[step['instruction'].translate(translator) for step in exp['utterances']] for exp in x_train]

act_exp = [[step['actions'] for step in exp['utterances']] for exp in x_train]

ins_exp[1]

act_exp[1]

#%%time
ins = [j.replace("'",'') for i in ins_exp for j in i]
act = [(['<start>']+j+['<end>']) for i in act_exp for j in i]


ins[6]

act[:2]

words_ins = [word for sentence in ins for word in (sentence.split())]

words_act = [word for sentence in act for word in (sentence)]

act_counter = Counter(words_act)
ins_counter = Counter(words_ins)

ins[1]

len(ins_counter.keys())

sns.distplot(np.log(list(ins_counter.values())),bins=20)

words_ins_cor = [x if ins_counter[x]>=3 else '<unknown>' for x in words_ins]

len(Counter(words_ins_cor))

vocab_set = set(words_ins_cor)
vocab = sorted(vocab_set)
vocab_dic = {}
for i, word in enumerate(vocab):
    vocab_dic[word]=i

vocab_out = sorted(set(words_act))

len(vocab_out)



########### Dynet

LAYERS = 1
INPUT_DIM = 50
HIDDEN_DIM = 100
STATE_SIZE = 32
ATTENTION_SIZE = 32
VOCAB_SIZE_input = len(vocab)
VOCAB_SIZE_out= len(vocab_out)


pc = dy.ParameterCollection()
encoder = dy.LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, pc)
decoder = dy.LSTMBuilder(LAYERS, INPUT_DIM+HIDDEN_DIM, HIDDEN_DIM, pc)
params_encoder={}
params_encoder["lookup"] = pc.add_lookup_parameters((VOCAB_SIZE_input, INPUT_DIM))
params_decoder= {}
params_decoder["lookup"] = pc.add_lookup_parameters((VOCAB_SIZE_out, INPUT_DIM))
params_decoder["R"] = pc.add_parameters((VOCAB_SIZE_out, HIDDEN_DIM))
params_decoder["bias"] = pc.add_parameters((VOCAB_SIZE_out))
################################
attention_w1 = pc.add_parameters( (ATTENTION_SIZE, STATE_SIZE*2))
attention_w2 = pc.add_parameters( (ATTENTION_SIZE, STATE_SIZE * LAYERS*2))
attention_v = pc.add_parameters( (1, ATTENTION_SIZE))
##################################

def do_one_sentence(encoder, decoder, params_encoder, params_decoder, sentence, output):
    # setup the sentence
    dy.renew_cg()
    s0 = encoder.initial_state()
    lookup = params_encoder["lookup"]

    sentence = ins[1]

    sentence = sentence.split()
    sentence = [vocab.index(c) if c in vocab else vocab.index('<unknown>') for c in sentence]
    s = s0
    loss = []
    generate = []
    ci = []

    for word in (sentence):
        s = s.add_input(lookup[word])
        ci.append(s.output())
    encode_output = s.output()

########################
#
#     w1 = dy.parameter(attention_w1)
#     input_mat = dy.concatenate_cols(vectors)
#     w1dt = None
# #############################


    _s0 = decoder.initial_state()
    _s = _s0
    R = params_decoder["R"]
    bias = params_decoder["bias"]
    index = 1
    input_word = "<start>"
    _lookup = params_decoder["lookup"]

    while True:
        # print(output[index])
        word = vocab_out.index(input_word)
        gt_y = vocab_out.index(output[index])
        # print((dy.concatenate([_lookup[word],encode_output])).npvalue().shape
        #     )
###############################
        #w1dt = w1dt or w1 * input_mat
        aji = []

        for cii in ci:
            if (_s.output() == None):
                aji.append(dy.dot_product(encode_output, cii))
            else:
                aji.append(dy.dot_product(_s.output(), cii))

        aj = dy.softmax(dy.concatenate(aji))
        wth = dy.concatenate_cols(ci)
        #assert (np.sum(aj.npvalue()) == 1)
        cj = wth * aj

        #_s = _s.add_input(dy.concatenate([_lookup[word], attend(input_mat, _s, w1dt)]))
#################################
        _s = _s.add_input(dy.concatenate([_lookup[word], cj]))
        # print((np.array(_s.output().value()).dot(R.as_array().T )).shape)
        probs = dy.softmax((R) * _s.output() + bias)
        prediction = np.argmax(probs.value())
        generate.append(vocab_out[prediction])
        loss.append(-dy.log(dy.pick(probs, gt_y)))
        if output[index] == '<end>':
            break
        index += 1
        input_word = vocab_out[prediction]

    loss = dy.esum(loss)
    return loss, generate




for i in range(10):
    print('Epoch%d' % i)
    count=0
    sum=0
    for sentence, output in zip(ins,act):
        count+=1
        trainer = dy.SimpleSGDTrainer(pc)
        loss,generate = do_one_sentence(encoder,decoder, params_encoder,params_decoder, sentence,output)
        loss_value = loss.value()
        loss.backward()
        trainer.update()
        sum+=loss_value
        if count % 2000 == 0:
            print("%.10f" % (sum/2000), end="\t")
            sum=0
        if count==1 or (count==6) or (count==7) :
            print(sentence,output,generate)

        #print(generate(rnn, params))