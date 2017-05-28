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