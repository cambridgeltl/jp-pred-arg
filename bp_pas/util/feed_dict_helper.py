import numpy as np

def sent2nump(sent):
    sent_ids = [arg_vocab.index(w.form) for w in sent]
    pases = [w.form for w in sent if w.is_prd]
    pred_ids = [pred_vocab.index(w.form) for w in sent if w.is_prd]
    label_mats = []
    for pas in sent.pas:
        lm = label_matrix(pas, arg_types, len(sent))
        label_mats.append(lm)
    lm = np.concatenate(label_mats, axis=1) #.shape
    return [sent_ids], [pred_ids], lm

def separate_pred_dicts(sent, pred_vocab, arg_vocab, arg_types, mask_token):
    dicts = []
    for pas in sent.pas:
        sent_ids = [arg_vocab.index(w.form) for w in sent]
        sent_ids[pas.pred.word_index] = arg_vocab.index(mask_token)
        pred_id = [pred_vocab.index(pas.pred.word_form)]
        pred_id_vector = [1.0 if pas.pred.word_index == i else 0.0 for i in range(len(sent))]
        labels =  label_matrix(pas, arg_types, len(sent))
#        dicts.append((sent_ids, pred_id, pred_id_vector, labels))
        dicts.append((sent_ids, pred_id, labels))
    return dicts

def label_matrix(pas, arg_types, sent_len):
    zeros = np.zeros((len(arg_types), sent_len))
    for arg in pas.args:
        zeros[arg_types.index(arg.arg_type), arg.word_index] = 1.0
    for j in range(sent_len):
        found = False
        for i in range(1, len(arg_types)):
            if zeros[i, j] == 1.0:
                found = True
        if not found:
            zeros[0,j] = 1.0
    return zeros





#def sent2ints(sent):
#    sent_ids = [vocab.index(w.form) for w in sent]
#    pas_ids = []
#    return sent_ids, pas_ids

#train_dicts = [sent2nump(td) for td in train_data]
#print(len(train_dicts[0][2].shape))
#print(len(partitioned_train_dicts[0][2].shape))