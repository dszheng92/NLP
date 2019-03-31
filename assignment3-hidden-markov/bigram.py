### Python 3
### Hidden Markov group created at 2018 April
### Cornell Tech

import numpy as np
from loader import Loader
from collections import defaultdict

class HiddenMarkov:
    def __init__(self, tags):
        self.states = list(tags)

    def train(self, words, label, method, weights = None):
        self.emissions = self._emissonprobability(words, label)
        self.transitions = self._transitionprobability(label, method, weights)

    def _emissonprobability(self, words, label):
        counts = defaultdict(int)
        numTags = defaultdict(int)
        emission = defaultdict(lambda: float('-inf'))
        for line, tags in zip(words, label):
            for word, tag in zip(line, tags):
                if tag == '*':
                    continue
                counts[(word, tag)] += 1
                numTags[tag] += 1
        k=0 #1
        for word, tag in counts.keys():
            emission[(word, tag)] = np.log((counts[(word, tag)] +k )/ (numTags[tag] + k * len(counts)))
        return emission

    def _transitionprobability(self, label, method, weights):
        if method == 'add1':
            transition = {}
            dependency = defaultdict(int)
            condition = defaultdict(int)
            for tags in label:
                for i, tag in enumerate(tags):
                    if tag == '*':
                        continue
                    current, prev = tags[i], tags[i - 1]
                    dependency[(current, prev)] += 1
                    condition[prev] += 1
            doubleTag = [(current, prev) for current in self.states
                            for prev in self.states]
            k=0.01
            for pairs in doubleTag:
                current, prev = pairs
                if (current, prev) in dependency:
                    transition[(current, prev)] = np.log((dependency[(current, prev)] + k) / (condition[prev] + k * len(condition)))
                else:
                    transition[(current, prev)] = np.log(k/ (condition[prev] +  k * len(condition)))
            return transition
        elif method == 'linear':
            assert weights is not None
            transition = defaultdict(float)
            dep1 = defaultdict(int)
            cond1 = defaultdict(int)
            dep0 = defaultdict(int)
            for tags in label:
                for i, tag in enumerate(tags):
                    if tag == '*':
                        continue
                    current, prev = tags[i], tags[i - 1]
                    dep1[(current, prev)] += 1
                    cond1[prev] += 1
                    dep0[current] += 1
            doublePair = [(current, prev) for current in self.states
                                for prev in self.states]
            for pair in doublePair:
                current, prev = pair
                if (current, prev) in dep1:
                    transition[(current, prev)] += weights[1] * dep1[(current, prev)] / cond1[prev]
                if current in dep0:
                    transition[(current, prev)] += weights[0] * dep0[current] / len(dep0)
                if (current, prev) not in transition:
                    transition[(current, prev)] = float('-inf')
                else:
                    transition[(current, prev)] = np.log(transition[(current, prev)])
        return transition


    def _linear_smooth(self, label, weights):
        #Linear interpolation smoothing.
        transition = defaultdict(float)
        dep1 = defaultdict(int)
        cond1 = defaultdict(int)
        dep0 = defaultdict(int)
        for tags in label:
            for i, tag in enumerate(tags):
                if tag == '*':
                    continue
                current, prev = tags[i], tags[i - 1]
                dep1[(current, prev)] += 1
                cond1[prev] += 1
                dep0[current] += 1
        doublePair = [(current, prev) for current in self.states
                            for prev in self.states]
        for pair in doublePair:
            current, prev = pair
            if (current, prev) in dep1:
                transition[(current, prev)] += weights[1] * dep1[(current, prev)] / cond1[prev]
            if current in dep0:
                transition[(current, prev)] += weights[0] * dep0[current] / len(dep0)
            if (current, prev) not in transition:
                transition[(current, prev)] = float('-inf')
            else:
                transition[(current, prev)] = np.log(transition[(current, prev)])
        return transition


    def beam_search(self, x, k):
        prediction = []
        for c, line in enumerate(x):
            trellis = [['*'] for _ in range(k)]
            total = [0] * k
            for i, word in enumerate(line):
                if word == '*':
                    continue
                topScore = [float('-inf')] * k
                pointer = [(0, '')] * k
                for index in range(k):
                    for state in self.states:
                        current = state
                        if self.emissions[(word, current)] == float('-inf'):
                            continue
                        if trellis[index][-1] == '':
                            continue
                        prev = trellis[index][-1]
                        score = self.emissions[(word, current)] + self.transitions[(current, prev)] + total[index]
                        minScore = min(topScore)
                        if score > minScore and score not in topScore and (index, current) not in pointer:
                            minPlace = topScore.index(minScore)
                            topScore[minPlace] = score
                            pointer[minPlace] = (index, current)
                newLine = []
                newScore = []
                for score, pp in zip(topScore, pointer):
                    index, current = pp
                    newLine.append(trellis[index] + [current])
                    newScore.append(score)
                trellis = newLine
                total = newScore
            maxScorePlace = total.index(max(total))
            prediction.append(trellis[maxScorePlace])
        return prediction

    def _viterbi(self, x):
        prediction = []
        for c, line in enumerate(x):
            prob = {state: float('-inf') for state in self.states}
            prob['*'] = 0
            back_pointer = {}
            back_pointer_idx = 0
            for word in line:
                if word == '*':
                    continue
                probNew = {state: float('-inf') for state in self.states}
                for current in self.states:
                    if self.emissions[(word, current)] == float('-inf'):
                        continue
                    for prev in self.states:
                        score = prob[prev] + self.transitions[(current, prev)] + self.emissions[(word, current)]
                        if score > probNew[current]:
                            probNew[current] = score
                            back_pointer[(back_pointer_idx, current)] = prev
                prob = probNew
                back_pointer_idx += 1
            prevState = max(prob, key=prob.get)
            lines = [prevState]
            back_pointer_idx -= 1
            while back_pointer_idx >= 0:
                current = lines[-1]
                tag = back_pointer[(back_pointer_idx, current)]
                lines.append(tag)
                back_pointer_idx -= 1
            prediction.append(lines[::-1])
        return prediction

    def inference(self, x, decode, k=None):
        ## Predict the tag with different mdecode mode
        if decode == 'beam':
            assert k is not None
            prediction = self.beam_search(x, k)
        elif decode == 'viterbi':
            prediction = self._viterbi(x)
        else:
            raise NotImplementedError('This type pf method is not implemented, try beam or viterbi')
        return prediction


    def suboptimal(self, x, y, decode, k=None):
        prediction = self.inference(x, decode, k)
        subNum = 0
        rightNum = 0
        idx = 0
        for line, predicted, golden in zip(x, prediction, y):
            predProb = 0
            goldProb = 0
            for i in range(len(line)):
                if golden[i] == '*':
                    continue
                predProb += (self.transitions[(predicted[i], predicted[i - 1])]
                              + self.emissions[(line[i], predicted[i])])
                goldProb += (self.transitions[(golden[i], golden[i - 1])]
                              + self.emissions[(line[i], golden[i])])
            if goldProb > predProb:
                subNum += 1
                print('[%d] Predicted' % idx)
                print(predicted)
                print('[%d] Gold' % idx)
                print(golden)
            if goldProb == predProb:
                rightNum += 1
            idx += 1
        return subNum / len(x), rightNum / len(x)

    def compAccu(self, dev_x, dev_y, decode, k=None):
        prediction = self.inference(dev_x, decode, k)
        correct = 0
        total = 0
        for pred, dev in zip(prediction, dev_y):
            for p, d in zip(pred, dev):
                if d == '*' or d == '<STOP>':
                    continue
                if p == d:
                    correct += 1
                total += 1
        return correct / total

    def submission(prediction, filename='bigram'):
        with open('./results/' + filename + '.csv', 'w') as Openfile:
            Openfile.write('id,tag\n')
            i = 0
            for sequences in prediction:
                for tt in sequences:
                    if tt == '*' or tt == '<STOP>':
                        continue
                    Openfile.write('{},"{}"\n'.format(i, tt))
                    i += 1


if __name__ == '__main__':
    loader = Loader(n_gram=2)
    words, label = loader.load_data('train')
    dev_x, dev_y = loader.load_data('dev')
    test_x, _ = loader.load_data('test')
    print('Data preprocessing done')
    markov = HiddenMarkov(tags=loader.tag_vocab)
    #markov.train(words, label, method='linear', weights = (0.65, 0.35))
    markov.train(words, label, method='add1')
    print('Training finished')
    #dev_acc = markov.compAccu(dev_x, dev_y, decode='viterbi')
    #print('Vitervi dev accuracy', dev_acc)
    dev_acc = markov.compAccu(dev_x, dev_y, decode='beam', k=4)
    print('Beam dev accuracy', dev_acc)

    #viterbi_sub, viterbi_correct = markov.suboptimal(dev_x, dev_y, decode='viterbi')
    #print('Viterbi suboptimal percentage', viterbi_sub)
    #print('Viterbi correct percentage', viterbi_correct)
    #beam_sub, beam_correct = markov.suboptimal(dev_x, dev_y, decode='beam', k=3)
    #print('Beam suboptimal percentage', beam_sub)
    #print('Beam correct percentage', beam_correct)
    #prediction = markov.inference(test_x, decode='viterbi')
    #submission(prediction, filename='trigram_add_one_viterbi')

