#Python3
### Hidden Markov group created at 2018 April
### Cornell Tech
import numpy as np
from loader import Loader
from collections import defaultdict


class HiddenMarkov:
    def __init__(self, tags):
        self.tag_pair = list(tags)
        self.states = [(current, prev, prev2) for current in self.tag_pair
                              for prev in self.tag_pair
                              for prev2 in self.tag_pair]

    def train(self, words, label, method, weights = None):
        self.emissions = self._emissonprobability(words, label)
        self.transitions = self._transitionprobability(label, method, weights)

    def _emissonprobability(self, words, label):
        emission = defaultdict(lambda: float('-inf'))
        counts = defaultdict(int)
        tag_count = defaultdict(int)
        for line, tags in zip(words, label):
            for word, tag in zip(line, tags):
                if tag == '*':
                    continue
                counts[(word, tag)] += 1
                tag_count[tag] += 1
        for word, tag in counts.keys():
            emission[(word, tag)] = np.log(counts[(word, tag)] / tag_count[tag])
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
                current, prev, prev2, prev3 = tags[i], tags[i - 1], tags[i - 2], tags[i - 3]
                dependency[(current, prev, prev2, prev3)] += 1
                condition[(prev, prev2, prev3)] += 1
        tag_quadro = [(current, prev, prev2, prev3) for current in self.tag_pair
                                  for prev in self.tag_pair
                                  for prev2 in self.tag_pair
                                  for prev3 in self.tag_pair]
        for quadro in tag_quadro:
            current, prev, prev2, prev3 = quadro
            if (current, prev, prev2, prev3) in dependency:
                transition[(current, prev, prev2, prev3)] = np.log((dependency[(current, prev, prev2, prev3)] + 1) / (condition[(prev, prev2, prev3)] + len(condition)))
            else:
                transition[(current, prev, prev2, prev3)] = np.log(1 / (condition[(prev, prev2, prev3)] + len(condition)))
        return transition

    def _linear_smooth(self, label, weights):
        transition = defaultdict(float)
        dep3 = defaultdict(int)
        cond3 = defaultdict(int)
        dep2 = defaultdict(int)
        cond2 = defaultdict(int)
        dep1 = defaultdict(int)
        cond1 = defaultdict(int)
        dep = defaultdict(int)
        for tags in label:
            for i, tag in enumerate(tags):
                if tag == '*':
                    continue
                current, prev, prev2, prev3 = tags[i], tags[i - 1], tags[i - 2], tags[i - 3]
                dep2[(current, prev, prev2, prev3)] += 1
                cond2[(prev, prev2, prev3)] += 1
                dep2[(current, prev, prev2)] += 1
                cond2[(prev, prev2)] += 1
                dep1[(current, prev)] += 1
                cond1[prev] += 1
                dep[current] += 1
        tag_quadro = [(current, prev, prev2, prev3) for current in self.tag_pair
                                  for prev in self.tag_pair
                                  for prev2 in self.tag_pair
                                  for prev3 in self.tag_pair]
        for quadro in tag_quadro:
            current, prev, prev2, prev3 = quadro
            if (current, prev, prev2, prev3) in dep3:
                transition[(current, prev, prev2, prev3)] += weights[3] * np.log(dep3[(current, prev, prev2, prev3)] / cond3[(prev, prev2, prev3)])
            if (current, prev, prev2) in dep2:
                transition[(current, prev, prev2)] += weights[2] * np.log(dep2[(current, prev, prev2)] / cond2[(prev, prev2)])
            if (current, prev) in dep1:
                transition[(current, prev, prev2)] += weights[1] * np.log(dep1[(current, prev)] / cond1[prev])
            if current in dep:
                transition[(current, prev, prev2)] += weights[0] * np.log(dep[current] / len(dep))
            if (current, prev, prev2) not in transition:
                transition[(current, prev, prev2)] = float('-inf')
        return transition


    def _beam_search(self, x, k):
        prediction = []
        for c, line in enumerate(x):
            trellis = [['*', '*', '*'] for _ in range(k)]
            total = [0] * k
            for i, word in enumerate(line):
                if word == '*':
                    continue
                topScore = [float('-inf')] * k
                pointer = [(0, '')] * k
                for index in range(k):
                    for state in self.states:
                        current, prev, prev2 = state
                        if prev2 != trellis[index][-2] or prev != trellis[index][-1] or self.emissions[(word, current)] == float('-inf'):
                            continue
                        prev3 = trellis[index][-3]
                        score = self.emissions[(word, current)] + self.transitions[(current, prev, prev2, prev3)] + total[index]
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
            max_score_idx = total.index(max(total))
            prediction.append(trellis[max_score_idx])
        return prediction

    def _viterbi(self, x):
        prediction = []
        for c, line in enumerate(x):
            prob = {state: float('-inf') for state in self.states}
            prob[('*', '*', '*')] = 0
            back_pointer = {}
            back_pointer_idx = 0
            for i, word in enumerate(line):
                if word == '*':
                    continue
                probNew = {state: float('-inf') for state in self.states}
                for current, prev, prev2 in self.states:
                    if self.emissions[(word, current)] == float('-inf'):
                        continue
                    for prev3 in self.tag_pair:
                        score = prob[(prev, prev2, prev3)] + self.transitions[(current, prev, prev2, prev3)] + self.emissions[
                            (word, current)]
                        if score > probNew[(current, prev, prev2)]:
                            probNew[(current, prev, prev2)] = score
                            back_pointer[(back_pointer_idx, current, prev)] = prev3
                prob = probNew
                back_pointer_idx += 1
            prevState = max(prob, key=prob.get)
            lines = list(prevState)
            back_pointer_idx -= 1
            while back_pointer_idx >= 0:
                current, prev, prev2 = lines[-3:]
                prev3 = back_pointer[(back_pointer_idx, current, prev, prev2)]
                lines.append(prev3)
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


    def suboptimal(self, x, y, decode, k = None):
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
                predProb += (self.transitions[(predicted[i], predicted[i - 1], predicted[i - 2],  predicted[i - 3])]
                              + self.emissions[(line[i], predicted[i])])
                goldProb += (self.transitions[(golden[i], golden[i - 1], golden[i - 2],  golden[i - 3])]
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


 # def error_analysis(self, Y_dev,Y_dev_predicted,dev_data, propername):
 #        index = [i for i in range(Y_dev.shape[0])]
 #        # build the confusion matrix
 #        confusion_matrix=[[0 for i in range(5)]for j in range(5)]
 #        for i in index:
 #            confusion_matrix[Y_dev[i]][Y_dev_predicted[i]]+=1
 #        print(np.array(confusion_matrix))
 #        print(np.array(confusion_matrix)/np.sum(np.array(confusion_matrix)))
 #        # print some error examples
 #        selected = np.where(Y_dev!=Y_dev_predicted)[0]
 #        selected = selected[:20]
 #        for i in selected:
 #            print("example ",i)
 #            print(dev_data[i])
 #            print("actual: ",propername.id_to_class[Y_dev[i]],"predicted:",propername.id_to_class[Y_dev_predicted[i]])

def submission(prediction, filename='4gram'):
    with open('./results/' + filename + '.csv', 'w') as f:
        f.write('id,tag\n')
        i = 0
        for sequences in prediction:
            for tt in sequences:
                if tt == '*' or tt == '<STOP>':
                    continue
                f.write('{},"{}"\n'.format(i, tt))
                i += 1

if __name__ == '__main__':
    loader = Loader(ngram = 4)
    words, label = loader.load_data('train')
    dev_x, dev_y = loader.load_data('dev')
    test_x, _ = loader.load_data('test')
    print('Data preprocessing done')

    markov = HiddenMarkov(tags=loader.tag_vocab)
    #HiddenMarkov.train(words, label, method='linear', weights=(0.5, 0.25, 0.15, 0.1))
    markov.train(words, label, method='add1')
    print('Training finished')


    #dev_acc = HiddenMarkov.compAccu(dev_x, dev_y, decode='viterbi')
    #print('DVitervi dev accuracy', dev_acc)
    dev_acc = markov.compAccu(dev_x, dev_y, decode='beam', k=3)
    print('Beam dev accuracy', dev_acc)


    #viterbi_sub, viterbi_correct = markov.suboptimal(dev_x, dev_y, decode='viterbi')
    #print('Viterbi suboptimal percentage', viterbi_sub)
    #print('Viterbi correct percentage', viterbi_correct)
    #beam_sub, beam_correct = markov.suboptimal(dev_x, dev_y, decode='beam', k=3)
    #print('Beam suboptimal percentage', beam_sub)
    #print('Beam correct percentage', beam_correct)

    #prediction = markov.inference(test_x, decode='viterbi')
    prediction = markov.inference(test_x, k = 3, decode='beam')
    submission(prediction, filename='4gram_add_one_beam')
    #submission(prediction, filename='4gram_linear_interpolate_beam')
    #submission(prediction, filename='4gram_linear_interpolate_viterbi')
    #submission(prediction, filename='4gram_add_one_viterbi')



