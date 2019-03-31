# NLP Final

This is the final examp of NLP. We applied bidirectional lstm with crf model to solve the Name Entity Recognition task.

## Getting Started

These are prerequisites you should have to run the experiment, and the parameters you can adjust to compare the results.

### Prerequisites

Pytorch

```
pip3 install torch torchvision
```

Glove

```
./data/glove.6B/glove.6B.300d.txt
```

### Parameters
```
parameters = OrderedDict()
parameters['char_lstm_dim'] = 100
parameters['char_embed_dim'] = 100
parameters['word_dim'] = 300
parameters['word_lstm_dim'] = 100
parameters['pre_emb'] = 'data/glove.6B/glove.6B.300d.txt'
parameters['cap_dim'] = 20
parameters['learning_rate'] = 0.001
parameters['epoch'] = 50
```

## Running the Experiment

```
python experiment.py
```


## Authors

* **Disheng Zheng** 
* **Xiaohang Lu** 

