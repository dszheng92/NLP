##Python 3.6
import numpy as np
import json
from collections import Counter
import matplotlib.pyplot as plt
import string
translator = str.maketrans('', '', string.punctuation)
import _dynet as dy
dyparams = dy.DynetParams()
dyparams.set_mem(1000)
dyparams.set_random_seed(666)
dyparams.set_autobatch(True)
dyparams.init()
import pandas as pd
import random
from sklearn.metrics import accuracy_score
from fsa import ExecutionFSA, EOS, ACTION_SEP, NO_ARG
from argparse import ArgumentParser
from model import Model
from fsa import WorldState
from alchemy_fsa import AlchemyFSA
from alchemy_world_state import AlchemyWorldState



def execute(world_state, action_sequence):
    """Executes an action sequence on a world state.
    Inputs:
        world_state (str): String representing an AlchemyWorldState.
        action_sequence (list of str): Sequence of actions in the format ["action arg1 arg2",...]
            (like in the JSON file).
    """
    alchemy_world_state = AlchemyWorldState(world_state)
    fsa = AlchemyFSA(alchemy_world_state)

    for action in action_sequence:
        split = action.split(" ")
        act = split[0]
        arg1 = split[1]
        if len(split) < 3:
            arg2 = NO_ARG
        else:
            arg2 = split[2]
        fsa.feed_complete_action(act, arg1, arg2)
    return fsa.world_state()


def one_hot_encode(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    return np.eye(n_classes)[x]
def get_state_embed3(one_state):
    """One hot encoding method: Generate order based encoding for one environment state - 168."""
    states = [i.split(':')[1] for i in one_state.split(' ')]
    colors = 'bgopry'
    state_vector = ['0','0','0','0','0','0','0']
    for i in range(7):
        for j in range(4):
            try:
                state = states[i][j]
                if state == '_':
                    state_vector[i] = state_vector[i] + '0'
                else:
                    state_vector[i] = state_vector[i] + str(colors.index(state))
            except IndexError:
                state_vector[i] = state_vector[i] + '0'
    vector = np.array([int(x) for x in ''.join(np.array([i[1:] for i in state_vector],dtype=str))])
    new_vector =[]
    for x in vector:
        if int(x)==0:
            y=np.zeros(6)
        else:
            y=one_hot_encode(int(x),6)
        new_vector.append(y)
    output = np.array([i for x in (new_vector) for i in x])
    return output


## Function to load data
def load(data):
    x_train = json.load(open(data))
    ins_exp = [[step['instruction'].translate(translator) for step in exp['utterances']] for exp in x_train]
    act_exp = [[step['actions'] for step in exp['utterances']] for exp in x_train]
    env_exp = [[step['after_env'] for step in exp['utterances']] for exp in x_train]
    init_env_exp = [np.tile(exp['initial_env'],len(exp['utterances'])) for exp in x_train]
    env_int = [(j) for i in init_env_exp for j in i]
    env =  [(j) for i in env_exp for j in i]
    ins = [j.replace("'",'') for i in ins_exp for j in i]
    act = [(['<start>']+j+['<end>']) for i in act_exp for j in i]
    id_exp = [exp['identifier'] for exp in x_train]
    return ins,act,env_int,id_exp


### Train on one sentence and get loss
def do_one_sentence(encoder,decoder, params_encoder,params_decoder, sentence,output,env,first,previous):
    pos_lookup = params_encoder["pos_lookup"]
    char_lookup = params_encoder["char_lookup"]
    char_v = params_decoder["attention_v"]
    char_w1 = params_decoder["attention_wc"]
    char_w2 = params_decoder["attention_bc"]
    sc_vector = []
    for i,world in enumerate(_state(env)):
        world=world
        sc0 = char_encoder.initial_state()
        sc=sc0
        for char in world:
            sc=sc.add_input(char_lookup[char2int[char]])
        sc_vector.append(dy.concatenate([sc.output(),pos_lookup[i]]))
    dy_sc_vector = dy.concatenate(sc_vector,d=1)
    s0 = encoder.initial_state()
    s = s0
    lookup = params_encoder["lookup"]
    attention_w = params_decoder["attention_w"]
    attention_b = params_decoder["attention_b"]
    sentence = sentence +' <end>'
    sentence = [vocab.index(c) if c in vocab else vocab.index('<unknown>') for c in sentence.split(' ')]
    loss = []
    generate = []
    s_vector=[]
    for word in (sentence):
        s = s.add_input(lookup[word])
        s_vector.append(dy.softmax(attention_w*s.output() + attention_b))
    encode_output = s.output()
    dy_s_vector = dy.concatenate(s_vector,d=1)
    _s0 = decoder.initial_state(s.s())
    _s = _s0
    R = params_decoder["R"]
    bias = params_decoder["bias"]
    index=1
    input_word = "<start>"
    _lookup = params_decoder["lookup"]
    while True:
        dy_env = dy.inputTensor(get_state_embed3(env))
        word = vocab_out.index(input_word)
        gt_y = vocab_out.index(output[index])
        
        weight = dy.softmax(dy.concatenate([dy.dot_product(x,_s.output()) for x in s_vector]))
        weight_char = dy.softmax(dy.concatenate([char_v*dy.tanh(char_w1*x + char_w2*_s.output()) for x in sc_vector]))
        
        encode_output = dy_s_vector * weight 
        encode_state = dy_sc_vector * weight_char 
        _s = _s.add_input(dy.concatenate([_lookup[word],encode_output,encode_state]))
        probs = dy.softmax((R) * _s.output() + bias)
        prediction = np.argsort(probs.npvalue())[-1]
        if (vocab_out[prediction])=='<start>':
            prediction = np.argsort(probs.npvalue())[-2]
        generate.append(vocab_out[prediction])
        loss.append( -dy.log(dy.pick(probs,gt_y)) )
        if output[index] == '<end>':
            break
        index+=1
        input_word = vocab_out[prediction]
        if input_word=='<end>':
            continue
        env = str(execute(env,[input_word]))
        if env=='None':
            env = '1:_ 2:_ 3:_ 4:_ 5:_ 6:_ 7:_'
    loss = dy.esum(loss)
    while '<start>' in generate: generate.remove('<start>')
    previous = s.output()
    return loss,generate,previous



### A generator function that do prediction on single instruction level
def generator(encoder,decoder, params_encoder,params_decoder, sentence,env,first,previous):
    pos_lookup = params_encoder["pos_lookup"]
    char_lookup = params_encoder["char_lookup"]
    char_v = params_decoder["attention_v"]
    char_w1 = params_decoder["attention_wc"]
    char_w2 = params_decoder["attention_bc"]
    sc_vector = []
    for i,world in enumerate(_state(env)):
        world=world
        sc0 = char_encoder.initial_state()
        sc=sc0
        for char in world:
            sc=sc.add_input(char_lookup[char2int[char]])
        sc_vector.append(dy.concatenate([sc.output(),pos_lookup[i]]))
    dy_sc_vector = dy.concatenate(sc_vector,d=1)
    s0 = encoder.initial_state()
    s = s0
    lookup = params_encoder["lookup"]
    attention_w = params_decoder["attention_w"]
    attention_b = params_decoder["attention_b"]
    sentence = sentence +' <end>'
    sentence = [vocab.index(c) if c in vocab else vocab.index('<unknown>') for c in sentence.split()]
    s_vector=[]
    generate = []
    for word in (sentence):
        s = s.add_input(lookup[word])
        s_vector.append(dy.softmax(attention_w*s.output() + attention_b))
    encode_output = s.output()
    dy_s_vector = dy.concatenate(s_vector,d=1)
    _s0 = decoder.initial_state(s.s())
    _s = _s0
    R = params_decoder["R"]
    bias = params_decoder["bias"]
    input_word = "<start>"
    _lookup = params_decoder["lookup"]
    repeat=0
    while True:
        dy_env = dy.inputTensor(get_state_embed3(env))
        repeat+=1
        word = vocab_out.index(input_word)
        weight = dy.softmax(dy.concatenate([dy.dot_product(x,_s.output()) for x in s_vector]))
        weight_char = dy.softmax(dy.concatenate([char_v*dy.tanh(char_w1*x + char_w2*_s.output()) for x in sc_vector]))
        encode_state = dy_sc_vector * weight_char 
        encode_output = dy_s_vector * weight 
        _s = _s.add_input(dy.concatenate([_lookup[word],encode_output,encode_state]))
        probs = dy.softmax((R) * _s.output() + bias)
        top=0
        while True:
            top+=1
            if top==50:
                top=1
                break
            prediction = np.argsort(probs.vec_value())[-top]
            if (vocab_out[prediction]=='<end>') : break
            if (vocab_out[prediction]=='<start>') : continue
            new_env = str(execute(env,[vocab_out[prediction]]))
            if new_env == 'None': continue
            break
        prediction = np.argsort(probs.vec_value())[-top]
        input_word = vocab_out[prediction]
        if input_word == '<end>':
            break
        if repeat>=10:
            break
        generate.append(input_word)
        env = str(execute(env,[input_word]))
        if env=='None':
            env = '1:_ 2:_ 3:_ 4:_ 5:_ 6:_ 7:_'
    while '<start>' in generate: generate.remove('<start>')
    previous = s.output()
    return generate,previous
    



### Function to predict given data
def predict(board_ins,board_init):
    act=[]
    count=0
    previous=None
    first=True
    for sentence,env in zip(board_ins,board_init): 
        if count%5 !=0:
            new_sentence = pre_sentence + ' <end> '+ sentence 
            new_env = str(execute(new_env,generate))
            if new_env=='None':
                new_env = '1:_ 2:_ 3:_ 4:_ 5:_ 6:_ 7:_'
        else:
            dy.renew_cg()
            new_sentence = sentence
            new_env = env
        generate,previous = generator(encoder,decoder, params_encoder,params_decoder, new_sentence,new_env,first,previous)
        act.append(generate)
        pre_sentence = sentence
        count+=1
        while '<end>' in generate: generate.remove('<end>')
    env_list = []
    final_env_list = []
    for i,env in enumerate(board_init):
        if i%5==0:
            new_env=env
        new_env = str(execute(new_env,act[i]))
        if new_env=='None':
            new_env = '1:_ 2:_ 3:_ 4:_ 5:_ 6:_ 7:_'
        env_list.append(new_env)
        if i%5==4:
            final_env_list.append(new_env)
    return env_list,final_env_list




def validate_train():  
    ### Function to generate train accuracy      
    train_pre_ins,train_pre_final_int = predict(ins,env_int)
    print(accuracy_score(train_pre_final_int,train_gt_int))
    print(accuracy_score(train_pre_ins,train_gt_ins))

def submit(id_name,states,filename='test_leaderboard_interaction_y.csv'):
    ### Function to generate submission file
    with open (filename,'w+') as Openfile:
        Openfile.writelines('id,final_world_state\n')
        for i,state in enumerate(states):
            Openfile.writelines('%s,%s\n' % (id_name[i],state)) 
        


# Load data
## Load training data and do pre-processing
ins,act,env_int,id_exp = load('train.json')
words_ins = [word for sentence in ins for word in (sentence.split())]
words_act = [word for sentence in act for word in (sentence)]
act_counter = Counter(words_act)
ins_counter = Counter(words_ins)
print ( 'Instruction word vocabulary size before unknown tagging: ',len(set(ins_counter)))
words_ins_cor = [x if ins_counter[x]>=10 else '<unknown>' for x in words_ins]
print ( 'Instruction word vocabulary size after unknown tagging: ',len(set(words_ins_cor)))

vocab_set = set(words_ins_cor +['<end>'])
vocab = sorted(vocab_set)
vocab_dic = {}
for i, word in enumerate(vocab):
    vocab_dic[word]=i
int2char = sorted(set(['g','o','r','b','p','y','_']))
char2int = {c:i for i,c in enumerate(int2char)}
def _state(world_state):
    return [x.split(':')[1] for x in world_state.split(' ')]
vocab_out = sorted(set(words_act))

## Load dev, leaderboard and final test data
board_ins,board_act,board_init,board_id = load('test_leaderboard.json')
dev_ins,dev_act,dev_init,dev_id = load('dev.json')
dev_gt_int = pd.read_csv('./dev_interaction_y.csv', index_col="id")['final_world_state'].values
dev_gt_ins = pd.read_csv('./dev_instruction_y.csv', index_col="id")['final_world_state'].values
train_gt_int = pd.read_csv('./train_interaction_y.csv', index_col="id")['final_world_state'].values
train_gt_ins = pd.read_csv('./train_instruction_y.csv', index_col="id")['final_world_state'].values
test_ins,test_act,test_init,test_id = load('test_final.json')
dev_gt_int = pd.read_csv('./dev_interaction_y.csv', index_col="id")['final_world_state'].values




# Dynet PART
## Initial SEQ2SEQ NETWORK 
LAYERS = 1
INPUT_DIM = 50
char_DIM = 20
HIDDEN_DIM = 100
ATTENTION_DIM=HIDDEN_DIM
VOCAB_SIZE_input = len(vocab)
VOCAB_SIZE_out= len(vocab_out)
VOCAB_char = len(int2char)

pc = dy.ParameterCollection()
encoder = dy.CompactVanillaLSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, pc)
decoder = dy.CompactVanillaLSTMBuilder(LAYERS, INPUT_DIM+HIDDEN_DIM*2, HIDDEN_DIM, pc)
params_encoder={}
params_encoder["lookup"] = pc.add_lookup_parameters((VOCAB_SIZE_input, INPUT_DIM))

params_decoder= {}
params_decoder["lookup"] = pc.add_lookup_parameters((VOCAB_SIZE_out, INPUT_DIM))
params_decoder["R"] = pc.add_parameters((VOCAB_SIZE_out, HIDDEN_DIM))
params_decoder["bias"] = pc.add_parameters((VOCAB_SIZE_out))
params_decoder["attention_w"] = pc.add_parameters((ATTENTION_DIM,HIDDEN_DIM))
params_decoder["attention_b"] = pc.add_parameters((ATTENTION_DIM))
params_decoder["attention_wc"] = pc.add_parameters((ATTENTION_DIM,HIDDEN_DIM))
params_decoder["attention_bc"] = pc.add_parameters((ATTENTION_DIM,HIDDEN_DIM))
params_decoder["attention_v"] = pc.add_parameters((1,ATTENTION_DIM))

char_encoder = dy.CompactVanillaLSTMBuilder(LAYERS, 50, 75, pc)
params_encoder["char_lookup"] = pc.add_lookup_parameters((VOCAB_char, 50))
params_encoder["pos_lookup"] = pc.add_lookup_parameters((7, 25))
dropout =0.05
encoder.set_dropouts(0, dropout)
decoder.set_dropouts(0, dropout)
char_encoder.set_dropouts(0, dropout)
trainer = dy.SimpleSGDTrainer(pc)



## TRAIN
### 200 epoch unless dev acc. on instruction above 0.58
dev_interaction = []
dev_instruction = []
for i in range(200):   
    print('Epoch%d' % i)
    count=0
    sum=0
    batch_loss=[]
    dy.renew_cg()
    previous=None
    first=True
    for sentence, output,env in zip(ins,act,env_int):
        if count%5 !=0:
            new_sentence = pre_sentence + ' <end> '+ sentence  
            new_env = str(execute(new_env,generate))
            if new_env=='None':
                new_env = '1:_ 2:_ 3:_ 4:_ 5:_ 6:_ 7:_'
        else:
            new_sentence = sentence
            new_env = env
        loss,generate,previous = do_one_sentence(encoder,decoder, params_encoder,params_decoder, new_sentence,output,new_env,first,previous)
        batch_loss.append(loss)
        sum+=loss.value()
        while '<end>' in generate : generate.remove('<end>')
        if len(batch_loss)>=5:
            losses = dy.average(batch_loss)
            losses.forward()
            losses.backward()
            trainer.update()
            batch_loss=[]
            dy.renew_cg()
        pre_sentence = sentence
        if count % 5000 == 4999:
            print("Loss: %.10f" % (sum/5000), end="\t")
            sum=0
        count+=1
    if batch_loss:
        losses = dy.average(batch_loss)
        losses.forward()
        losses.backward()
        trainer.update()
        dy.renew_cg()
    dev_pre_ins,dev_pre_final_int = predict(dev_ins,dev_init)
    dev_int_acc  = accuracy_score(dev_pre_final_int,dev_gt_int)
    dev_ins_acc  = accuracy_score(dev_pre_ins,dev_gt_ins)
    dev_interaction.append(dev_int_acc )
    dev_instruction.append(dev_ins_acc )
    print('\nDev. interaction acc: ',dev_int_acc,'\nDev. instruction acc.: ',dev_ins_acc) 
    if dev_ins_acc>0.58:
        break

## Predict and submit
validate_train()
board_pre_ins,board_pre_final_int = predict(board_ins,board_init)
submit(board_id,board_pre_final_int,filename='test_leaderboard_interaction_y.csv')
test_pre_ins,test_pre_final_int = predict(test_ins,test_init)
submit(test_id,test_pre_final_int,'test_final_interaction_y.csv')

