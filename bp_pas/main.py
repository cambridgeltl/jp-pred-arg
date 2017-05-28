
import bp_pas.params as params
import bp_pas.corpus as corpus
import bp_pas.stats as stats
import sys
import gflags
#from bp_pas.eval import SampleEval, ResultEval, ntc_sent2conll
from bp_pas.eval import evaluate


def main(flags):
    print(flags.train_data)
    ntc = corpus.NTCLoader()
    train_data = ntc.load_corpus(flags.train_data, flags.max_train_instances)
    test_data  = ntc.load_corpus(flags.test_data, flags.max_train_instances)
    dev_data   = ntc.load_corpus(flags.dev_data, flags.max_train_instances)
    print('{} train docs.'.format(len(train_data)))
    print('{} test docs.'.format(len(test_data)))
    print('{} dev docs.'.format(len(dev_data)))
    stats.corpus_statistics(train_data)
    stats.show_case_dist(train_data)

    # for w in train_data[0][0]:
    #     print(w)
    #     print(w.chunk_index)
    #     print(w.chunk_head)
    #     print(w.is_prd)
    #     print(w.arg_indices)
    #     print(w.arg_types)
    #     print()

#    print(test_data[0])
#    print(test_data[0][0])

#    re = SampleEval()
#    for d1, d2 in zip(test_data, test_data):
#        result = d1
#        sample = d2
#        re.update_results(y_sys_batch=result, sample=sample)
    test = [pas for doc in test_data for sent in doc for pas in sent.pas]
    gold = [pas for doc in test_data for sent in doc for pas in sent.pas]

    evaluate(test, gold)

#    for d1, d2 in zip(test_data, test_data):
#        print(d1)
#        print(len(d1))
#        print(d1[0])
#        s1 = ntc_sent2conll(d1)
#        print(s1)
#        s2 = ntc_sent2conll(d2)
#        re._add_results_gold(s1)
#        re._add_results_sys(s2)
#        break



if __name__ == "__main__":
    params.load_defaults()
    flags = gflags.FLAGS
    argv = flags(sys.argv)
    main(flags)
