import gflags
import time
import random

def load_defaults():
    ### Experiment Parameters
    gflags.DEFINE_integer('num_experiments', 1, 'the number of times to train the model (with different initialization)')
    gflags.DEFINE_integer('seed', 1, 'a random seed used in all randomized model initialization')

    ### Data Parameters
    gflags.DEFINE_string('train_data', 'data/NTC_1.5/processed/train.utf8', 'the path to the train data file')
    gflags.DEFINE_string('test_data', 'data/NTC_1.5/processed/test.utf8', 'the path to the test data file')
    gflags.DEFINE_string('dev_data', 'data/NTC_1.5/processed/dev.utf8', 'the path to the dev data file')
    gflags.DEFINE_integer('max_train_instances', 100000, 'max number of instances to read for each train/test/dev set')

    ### Model Parameters
    ## Representation
    gflags.DEFINE_boolean('use_context', True, 'construct the representation from LSTM context embeddings')
    gflags.DEFINE_boolean('use_sp', True, 'construct the representation directly from pred/arg embeddings')
    gflags.DEFINE_integer('context_dims', 20, 'output dimension for LSTM context embeddings')
    gflags.DEFINE_integer('sp_dims', 8, 'dimensionality of arg embedding for selectional preference')
    gflags.DEFINE_integer('num_context_layers', 2, 'number of layers in the context LSTM')
    gflags.DEFINE_float('context_dropout', 0.0, 'amount of dropout on each LSTM layer in the context representation')