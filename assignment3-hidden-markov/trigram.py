
### Python 3
### Hidden Markov group created at 2018 April
### Cornell Tech
import numpy as np
from loader import Loader
from collections import defaultdict


class HiddenMarkov:
    def __init__(self, tags):
        self.tag_pair = list(tags)
        self.states = [(current, prev) for current in self.tag_pair
                              for prev in self.tag_pair]

    def train(self, words, label, method, weights = None):
        self.transitions = self._transitionprobability(label, method, weights)

        self.emissions = self._emissonprobability(words, label)
        

    def _emissonprobability(self, words, label):
        emission = defaultdict(lambda: float('-inf'))
        counts = defaultdict(int)
        numTags = defaultdict(int)
        for line, tags in zip(words, label):
            for word, tag in zip(line, tags):
                if tag == '*':
                    continue
                counts[(word, tag)] += 1
                numTags[tag] += 1
        for word, tag in counts.keys():
            emission[(word, tag)] = np.log(counts[(word, tag)] / numTags[tag])
            #emissions[(word, tag)] = np.log((counts[(word, tag)] + 0.05)/ (numTags[tag]+ 0.05 * len(counts)))
        return emission

    def _transitionprobability(self, label, method, weights):
        if method == 'add1':
            return self._add_one_smooth(label)
        elif method == 'linear':
            assert weights is not None
            return self._linear_smooth(label, weights)

    def _add_one_smooth(self, label):
        transition = {}
        dependency = defaultdict(int)
        condition = defaultdict(int)
        for tags in label:
            for i, tag in enumerate(tags):
                if tag == '*':
                    continue
                current, prev, prev2 = tags[i], tags[i - 1], tags[i - 2]
                dependency[(current, prev, prev2)] += 1
                condition[(prev, prev2)] += 1
        threetag = [(current, prev, prev2) for current in self.tag_pair
                                  for prev in self.tag_pair
                                  for prev2 in self.tag_pair]
        k = 0.02
        for tri in threetag:
            current, prev, prev2 = tri
            if (current, prev, prev2) in dependency:
                transition[(current, prev, prev2)] = np.log((dependency[(current, prev, prev2)] + k) / (condition[(prev, prev2)] + k* len(condition)))
            else:
                transition[(current, prev, prev2)] = np.log(k/ (condition[(prev, prev2)] + k * len(condition)))
        return transition


    ## smoothing method: linear interpolation
    def _linear_smooth(self, label, weights):
        transition = defaultdict(float)
        cond2 = defaultdict(int)
        dep2 = defaultdict(int)
        cond1 = defaultdict(int)
        dep1 = defaultdict(int)
        cond0 = defaultdict(int)
        for tags in label:
            for i, tag in enumerate(tags):
                if tag == '*':
                    continue
                current, prev, prev2 = tags[i], tags[i - 1], tags[i - 2]
                cond2[(current, prev, prev2)] += 1
                dep2[(prev, prev2)] += 1
                cond1[(current, prev)] += 1
                dep1[prev] += 1
                cond0[current] += 1
        trigroup = [(current, prev, prev2) for current in self.tag_pair
                                  for prev in self.tag_pair
                                  for prev2 in self.tag_pair]
        for tri in trigroup:
            current, prev, prev2 = tri
            if (current, prev, prev2) in cond2:
                transition[(current, prev, prev2)] += weights[2] * np.log(cond2[(current, prev, prev2)] / dep2[(prev, prev2)])
            if (current, prev) in cond1:
                transition[(current, prev, prev2)] += weights[1] * np.log(cond1[(current, prev)] / dep1[prev])
            if current in cond0:
                transition[(current, prev, prev2)] += weights[0] * np.log(cond0[current] / len(cond0))
            if (current, prev, prev2) not in transition:
                transition[(current, prev, prev2)] = float('-inf')
        return transition


    def _beam_search(self, x, k):
        prediction = []
        for c, line in enumerate(x):
            trellis = [['*', '*'] for _ in range(k)]
            total = [0] * k
            for i, word in enumerate(line):
                if word == '*':
                    continue
                topScore = [float('-inf')] * k
                pointer = [(0, '')] * k
                for index in range(k):
                    for state in self.states:
                        current, prev = state
                        if prev != trellis[index][-1] or self.emissions[(word, current)] == float('-inf'):
                            continue
                        prev2 = trellis[index][-2]
                        score = self.emissions[(word, current)] + self.transitions[(current, prev, prev2)] + total[index]
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
            prob[('*', '*')] = 0
            back_pointer = {}
            back_pointer_idx = 0
            for i, word in enumerate(line):
                if word == '*':
                    continue
                probNew = {state: float('-inf') for state in self.states}
                for current, prev in self.states:
                    if self.emissions[(word, current)] == float('-inf'):
                        continue
                    for prev2 in self.tag_pair:
                        score = prob[(prev, prev2)] + self.transitions[(current, prev, prev2)] + self.emissions[(word, current)]
                        if score > probNew[(current, prev)]:
                            probNew[(current, prev)] = score
                            back_pointer[(back_pointer_idx, current, prev)] = prev2
                prob = probNew
                back_pointer_idx += 1
            prevState = max(prob, key=prob.get)
            lines = list(prevState)
            back_pointer_idx -= 1
            while back_pointer_idx >= 0:
                current, prev = lines[-2:]
                prev2 = back_pointer[(back_pointer_idx, current, prev)]
                lines.append(prev2)
                back_pointer_idx -= 1
            prediction.append(lines[::-1])
        return prediction

    def inference(self, x, decode, k=None):
        """Tags a sequence with part of speech tags.

                You should implement different kinds of inference (suggested as separate
                methods):

                    - greedy decoding
                    - decoding with beam search
                    - viterbi
        """
        if decode == 'beam':
            assert k is not None
            prediction = self._beam_search(x, k)
        elif decode == 'viterbi':
            prediction = self._viterbi(x)
        else:
            raise NotImplementedError('Method not implemented')
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
                predProb += (self.transitions[(predicted[i], predicted[i - 1], predicted[i - 2])]
                              + self.emissions[(line[i], predicted[i])])
                goldProb += (self.transitions[(golden[i], golden[i - 1], golden[i - 2])]
                              + self.emissions[(line[i], golden[i])])
            if goldProb == predProb:
                rightNum += 1
            if goldProb > predProb:
                subNum += 1
                print('[%d] Predicted' % idx)
                print(predicted)
                print('[%d] Golden' % idx)
                print(golden)
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

def submission(prediction, filename='trigram'):
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
    loader = Loader(n_gram=3)
    words, label = loader.load_data('train')
    dev_x, dev_y = loader.load_data('dev')
    test_x, _ = loader.load_data('test')
    print('Data preprocessing done')

    markov = HiddenMarkov(tags=loader.tag_vocab)
    #HiddenMarkov.train(words, label, method='linear', weights=(0.6, 0.25, 0.15))
    markov.train(words, label, method='add1')
    print('Training finished')

    #dev_acc = HiddenMarkov.compAccu(dev_x, dev_y, decode='viterbi')
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
    #prediction = markov.inference(test_x, k = 3, decode='beam')
    #submission(prediction, filename='trigram_add_one_beam')
    #submission(prediction, filename='trigram_linear_interpolate_beam')
    #submission(prediction, filename='trigram_linear_interpolate_viterbi')
    #submission(prediction, filename='trigram_add_one_viterbi')
