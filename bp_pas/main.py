
# import bp_pas.params as params
# import bp_pas.corpus as corpus
# import bp_pas.stats as stats
# import sys
# import gflags
# from bp_pas.eval import evaluate

import bp_pas.params as params
import bp_pas.corpus as corpus
import bp_pas.stats as stats
from bp_pas.eval import evaluate
from bp_pas.util.vocab import Vocab

import sys
import gflags
import tensorflow as tf
import numpy as np

from bp_pas.eval import evaluate
from bp_pas.ling.pas import PAS, Predicate, Argument
from bp_pas.ling.sent import Sentence

from bp_pas.model.baseline import Model
from bp_pas.util.feed_dict_helper import separate_pred_dicts


def main(flags):
    # tmp params to add to flags
    unk_threshold = 1
    unk_token = '<UNK>'

    if not flags.use_context:
        flags.context_dims = 0
    if not flags.use_sp:
        flags.sp_dims = 0

    synt_arg_embedding_size = flags.context_dims #+ 1
    sem_arg_embedding_size = flags.sp_dims

    complete_embedding_size = synt_arg_embedding_size + sem_arg_embedding_size # + 1 # for the pred id concat
    num_training_instances = 1

    assert complete_embedding_size % 2 == 0
    print(complete_embedding_size)

    # Load data
    print(flags.train_data)
    ntc = corpus.NTCLoader()
    train_data = ntc.load_corpus(flags.train_data, flags.max_train_instances)
    test_data = ntc.load_corpus(flags.test_data, flags.max_train_instances)
    dev_data = ntc.load_corpus(flags.dev_data, flags.max_train_instances)
    print('{} train sentences.'.format(len(train_data)))
    print('{} test sentences.'.format(len(test_data)))
    print('{} dev sentences.'.format(len(dev_data)))
    stats.corpus_statistics(train_data)
    stats.show_case_dist(train_data)

    # Unpack data from "docs"->sents
    train_data = train_data[0]
    test_data = test_data[0]
    dev_data = dev_data[0]
    print('{} train sentences.'.format(len(train_data)))
    print('{} test sentences.'.format(len(test_data)))
    print('{} dev sentences.'.format(len(dev_data)))

    # Setup vocabulary
    all_sents = train_data + test_data + dev_data
    print('Number of total sentences: {}'.format(len(all_sents)))

    print('Collecting all tokens...')
    word_tokens = list(word.form
                       for sent in all_sents
                       for word in sent)
    print('Total tokens: {}\n'.format(len(word_tokens)))

    print('Building (argument) vocabulary...')
    arg_vocab = Vocab(init_words=word_tokens,
                      unk_token=unk_token,
                      unk_threshold=unk_threshold)
    arg_vocab.freeze()
    arg_vocabulary_size = len(arg_vocab)
    print('Vocabulary size: {}\n'.format(arg_vocabulary_size))

    print('Collecting predicates...')
    pred_vocab = Vocab(init_words=[word.form
                                   for sent in all_sents
                                   for word in sent if word.is_prd])
    pred_vocab.freeze()
    pred_vocabulary_size = len(pred_vocab)
    print('Number of predicates: {}\n'.format(pred_vocabulary_size))

    print('Collecting argument types...')
    arg_types = ['NIL'] + list(set([arg.arg_type
                                    for sent in all_sents
                                    for pas in sent.pas
                                    for arg in pas.args]))
    print('Arg types: ', arg_types)
    num_types = len(arg_types)

    # print(pred_vocab.elems())

    # Convert data structures into numpy ndarrays for placeholders
    train_dicts = [ex
                   for td in train_data
                   for ex in separate_pred_dicts(td, pred_vocab, arg_vocab, arg_types)]

    test_dicts = [separate_pred_dicts(td, pred_vocab, arg_vocab, arg_types)
                  for td in train_data]

    # Just for now
    test_data = train_data
    print(len(train_dicts))
    print(len(test_dicts))
    print('pred ids: ', train_dicts[0][2])
    print('pred ids: ', len(train_dicts[0][2]))
    print('sent ids: ', len(train_dicts[0][0]))
#    assert (len(sent_ids) == len(pred_id_vector))

    # Setup the context embedding
    tf.reset_default_graph()

    # Construct syntactic embedding matrix for contex/arguments
    synt_arg_embeddings = tf.Variable(
        tf.random_uniform([arg_vocabulary_size, synt_arg_embedding_size], -1.0, 1.0, dtype=tf.float64), trainable=False)

    # Construct semantic embedding matrix for selectional preference
    sem_arg_embeddings = tf.Variable(
        tf.random_uniform([arg_vocabulary_size, sem_arg_embedding_size], -1.0, 1.0, dtype=tf.float64), trainable=False)

    # Construct embedding matrix for predicates
    pred_embeddings = tf.Variable(
        tf.random_uniform([pred_vocabulary_size, num_types * complete_embedding_size], -1.0, 1.0, dtype=tf.float64),
        trainable=True)

    # Setup placeholders
    batch_size = 1
    sent_placeholder = tf.placeholder(tf.int32, shape=[None])
    pred_placeholder = tf.placeholder(tf.int32, shape=[None])
    gold_placeholder = tf.placeholder(tf.float64, shape=[num_types, None])
    pred_id_placeholder = tf.placeholder(tf.float64, shape=[None])


    model = Model(pred_embeddings, synt_arg_embeddings, sem_arg_embeddings,
                  sent_placeholder, pred_placeholder, gold_placeholder, flags)

    rep = model.complete_embeddings(sent_placeholder, pred_id_placeholder, sem_arg_embeddings, synt_arg_embeddings, synt_arg_embedding_size)
    scored_mat_opt = model.pred_full_scoring(pred_placeholder, pred_embeddings, rep)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    loss = model.loss_old(gold_placeholder, pred_placeholder, pred_embeddings, rep)
    opt_op = opt.minimize(loss)

    def meta_decode(sess, instance):
        #    print('In decode')
        # Collect model-scores labels-x-tokens mats
        int_inputs = separate_pred_dicts(instance, pred_vocab, arg_vocab, arg_types)
        #    print('plen: ', len(int_inputs))
        scored_mats = []
        for sent_ids, pred_ids, pred_id_vector, gold_labels in int_inputs:
            print('pred ids: ', pred_id_vector)
            print('sent ids: ', sent_ids)
            assert(len(sent_ids) == len(pred_id_vector))
            feed_dict = {
                sent_placeholder: sent_ids,
                pred_placeholder: pred_ids,
                gold_placeholder: gold_labels,
                pred_id_placeholder: pred_id_vector
            }
            scored_mats.append(sess.run(scored_mat_opt, feed_dict=feed_dict))
            #        print(gold_labels)
            #        print('\n')
            #        print(scored_mats[-1])
            #        print('------------------------')
            #
            #    print('scored ', len(scored_mats))
        pred_indices = [i for i, w in enumerate(instance.words) if w.is_prd]
        num_preds = len(pred_indices)
        pases = []
        for pred_id, scored_mat in zip(pred_indices, scored_mats):
            #        print(pred_id)
            pred = Predicate(word_index=pred_id,
                             word_form=instance.words[pred_id].form)
            args = []
            for slot, slot_id in zip(arg_types, range(num_types)):
                if slot != 'NIL':
                    max_index = np.argmax(scored_mat[slot_id])
                    #                print(slot, ': ', max_index)
                    args.append(Argument(word_index=max_index,
                                         word_form=instance.words[max_index].form,
                                         arg_type=slot))
                    #                print('\t', max_index)
            pases.append(PAS(pred, args))
            #        print('psize: ', len(pases))
        return Sentence(instance.words, pases)

    num_epochs = 10000
    for epoch in range(num_epochs):
        epoch_loss = 0
        for train_dict in train_dicts:
            sent_ids, pred_ids, pred_id_vector, gold_labels = train_dict
            #        print(gold_labels.shape)
            _, closs = sess.run([opt_op, loss], feed_dict={
                sent_placeholder: sent_ids,
                pred_placeholder: pred_ids,
                pred_id_placeholder: pred_id_vector,
                gold_placeholder: gold_labels
            })
            epoch_loss += closs.mean()
        if epoch > 0 and epoch % 1000 == 0:
            decoded = [pas
                       for tsent in train_data
                       for pas in meta_decode(sess, tsent).pas]  # [0:1]
            #        decoded = [pas
            #                   for tsent, tdict in zip(train_data, train_dicts)
            #                   for pas in meta_decode(sess, tsent)] # #model.decode(sess, tsent, tdict).pas]
            gold = [pas
                    for sent in train_data
                    for pas in sent.pas]  # [0:1]
            evaluate(decoded, gold, verbose=False)
            print('{0:>4}: {1:.3f}'.format(epoch, epoch_loss))
















                #
    # print(flags.train_data)
    # ntc = corpus.NTCLoader()
    # train_data = ntc.load_corpus(flags.train_data, flags.max_train_instances)
    # test_data  = ntc.load_corpus(flags.test_data, flags.max_train_instances)
    # dev_data   = ntc.load_corpus(flags.dev_data, flags.max_train_instances)
    # print('{} train docs.'.format(len(train_data)))
    # print('{} test docs.'.format(len(test_data)))
    # print('{} dev docs.'.format(len(dev_data)))
    # stats.corpus_statistics(train_data)
    # stats.show_case_dist(train_data)

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

#    test = [pas for doc in test_data for sent in doc for pas in sent.pas]
#    gold = [pas for doc in test_data for sent in doc for pas in sent.pas]
#    evaluate(test, gold)

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
