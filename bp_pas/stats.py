import numpy as np


def show_case_dist(corpus):
    print('\nCASE DISTRIBUTION')
    case_types = np.zeros((3, 5))
    n_prds = 0

    for doc in corpus:
        for sent in doc:
            for w in sent:
                if w.is_prd is False:
                    continue
                flag = False
                for case_index, arg_type in enumerate(w.arg_types):
                    if arg_type > -1:
                        case_types[case_index][arg_type] += 1
                        if 0 < arg_type < 3:
                            flag = True
                if flag:
                    n_prds += 1

    for case_index, cases in enumerate(case_types):
        if case_index == 0:
            label = 'Ga'
        elif case_index == 1:
            label = 'O'
        else:
            label = 'Ni'
        print('\t{}\tBST: {:.0f}  DEP: {:.0f}  INTRA-ZERO: {:.0f}  INTER-ZERO: {:.0f}  EXOPHORA: {:.0f}'.format(
            label, cases[0], cases[1], cases[2], cases[3], cases[4]))
    print('\n\tPredicates: {}'.format(n_prds))


def corpus_statistics(corpus):
    if corpus is None:
        return

    print('\nCORPUS STATISTICS')

    """
    NAIST Ver. 1.5; DOC Train:1751, Dev:480, Test:696
    """
    n_sents = 0
    n_words = 0
    n_pds = 0
    n_args = 0

    for doc in corpus:
        n_sents += len(doc)
        for sent in doc:
            n_words += len(sent)
            for word in sent.words:
                if word.is_prd:
                    n_pds += 1
                    for case_i, arg_id in enumerate(word.arg_ids):
                        if arg_id > -1:
                            n_args += 1

    print('\tDocs: {}  Sents: {}  Words: {}'.format(len(corpus), n_sents, n_words))
    print('\tPredicates: {}  Arguments {}'.format(n_pds, n_args))
    print('\n')


def sample_statistics(samples, vocab_label):
    if samples is None:
        return

    print('\nSAMPLE STATISTICS')

    """
    The case distribution does not match with that of corpus_statistics(),
    because one word sometimes plays multiple case roles.
    Even in such cases, we assign one case role for a word.
    """

    n_samples = 0
    n_args = 0

    label_count = {}
    for key in vocab_label.w2i.keys():
        label_count[key] = 0
    n_labels = vocab_label.size()

    for sample in samples:
        sent = sample.label_ids
        for prd_labels in sent:
            flag = False
            for label in prd_labels:
                label_count[vocab_label.get_word(label)] += 1
                if 0 < label < n_labels-1:
                    n_args += 1
                    flag = True
            if flag:
                n_samples += 1

    print('\tSamples: {}'.format(n_samples))
    print('\t'),
    for case, count in label_count.items():
        print('{}: {}  '.format(case, count))
    print('\n')
