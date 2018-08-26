# assignment4-master-alchemist

# How to run

 ```sh   
    python seq2seq.py
```


### Parameters such as:

Batch size, Bulder, Optimizer. 

Number of epoch.

State embedding and attention.

### can be selected and modified in each file (in the commented out section).


Utility scripts contained in this directory:

* alchemy_fsa.py: defines an FSA used to execute action sequences on a world state.
* alchemy_world_state.py: class for world states in the Alchemy domain that operate as a sequence of stacks.
* evaluate.py: script for evaluating CSV predicted states with CSV labels.
* extract_labels.py: used for generating the state labels for the training and development data.
* fsa.py: abstract class for FSA and world states.

Data files in this directory:

* train.json, dev.json: contain the labeled training and development data.
* test.json: contains the unlabeled test inputs.
* results/test_interaction_y.csv: placeholder for your predictions on the test set (interaction-level predictions).
