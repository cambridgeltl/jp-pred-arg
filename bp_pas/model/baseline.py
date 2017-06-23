import tensorflow as tf
from bp_pas.util.return_tensor import ReturnTensor

class Model:

    def __init__(self, synt_pred_embeddings, synt_arg_embeddings,
                       sem_pred_embeddings, sem_arg_embeddings,
                       sent_placeholder, pred_placeholder, gold_placeholder, flags):
        self.synt_pred_embeddings = synt_pred_embeddings
        self.synt_arg_embeddings = synt_arg_embeddings
        self.sem_pred_embeddings = sem_pred_embeddings
        self.sem_arg_embeddings = sem_arg_embeddings
        self.sent_placeholder = sent_placeholder
        self.pred_placeholder = pred_placeholder
        self.gold_placeholder = gold_placeholder
        self.flags = flags



    def syntactic_embeddings(self, sent, output_dim):
        # Shape [batch_size, max_sent_len, emb_dim]
        embed_rep = tf.gather(self.synt_arg_embeddings, sent)
#        embed_rep = tf.reshape(embed_rep, [1, -1, 50])
        embed_rep = tf.expand_dims(embed_rep, 0)
#        embed_rep = tf.expand_dims(embed_rep, 1)
        # List of length max_sent_len, comprising [batch_size, emb_dim] tensors
        fw_cell = tf.contrib.rnn.LSTMCell(num_units=output_dim / 2, state_is_tuple=True)
        bw_cell = tf.contrib.rnn.LSTMCell(num_units=output_dim / 2, state_is_tuple=True)

        # Construct RNN cells for all layers
        num_units  = output_dim / 2
        dropout = self.flags.context_dropout
        fw_single_cells = []
        bw_single_cells = []

        for _ in range(self.flags.num_context_layers):
            #            fw_single_cell =  tf.contrib.rnn.LayerNormBasicLSTMCell(num_units) #tf.contrib.rnn.GRUCell(num_units)  # Or LSTMCell(num_units)
            fw_single_cell = tf.contrib.rnn.LSTMCell(num_units,
                                                     state_is_tuple=True)  # tf.contrib.rnn.GRUCell(num_units)  # Or LSTMCell(num_units)
            do_fw_single_cell = tf.contrib.rnn.DropoutWrapper(fw_single_cell, output_keep_prob=1.0 - dropout)
            #            bw_single_cell =  tf.contrib.rnn.LayerNormBasicLSTMCell(num_units) # tf.contrib.rnn.GRUCell(num_units)  # Or LSTMCell(num_units)
            bw_single_cell = tf.contrib.rnn.LSTMCell(num_units,
                                                     state_is_tuple=True)  # tf.contrib.rnn.GRUCell(num_units)  # Or LSTMCell(num_units)
            do_bw_single_cell = tf.contrib.rnn.DropoutWrapper(bw_single_cell, output_keep_prob=1.0 - dropout)
            fw_single_cells.append(do_fw_single_cell)
            bw_single_cells.append(do_bw_single_cell)
        # Add into multi cell
        fw_cell = tf.contrib.rnn.MultiRNNCell(fw_single_cells)
        bw_cell = tf.contrib.rnn.MultiRNNCell(bw_single_cells)

        #        return embed_rep
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell,
            cell_bw=bw_cell,
            dtype=tf.float64,
#            time_major=True,
            #        sequence_length=X_lengths,
            inputs=embed_rep)
#        output_fw, output_bw = outputs
#        states_fw, states_bw = states

        # Returns [N, output_dim]
        final_rep = tf.squeeze(tf.concat(outputs, 2))
        return final_rep, embed_rep

    def loss_even_newer(self, gold, prediction):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.transpose(prediction),
                                                       labels=tf.transpose(gold))
        return loss

    def new_pred_full_scoring(self, sent, preds):
        # Shape [N, embedding_sizes]
        sem_embed_mat  = tf.gather(self.sem_arg_embeddings, sent)
        synt_embed_mat, debug_embed_rep = self.syntactic_embeddings(sent, output_dim=self.flags.context_dims)


        sem_pred_mat = tf.squeeze(tf.gather(self.sem_pred_embeddings, preds))
        sem_pred_mat = tf.reshape(sem_pred_mat, shape=[tf.shape(sem_embed_mat)[-1], -1])

        synt_pred_mat = tf.squeeze(tf.gather(self.synt_pred_embeddings, preds))
        synt_pred_mat = tf.reshape(synt_pred_mat, shape=[tf.shape(synt_embed_mat)[-1], -1])

#        synt_scores =  tf.log(tf.nn.relu(tf.transpose(tf.matmul(synt_embed_mat, synt_pred_mat))))
        synt_scores =  tf.transpose(tf.matmul(synt_embed_mat, synt_pred_mat))
        sem_scores  =  tf.log(tf.transpose(tf.nn.softmax(tf.matmul(sem_embed_mat, sem_pred_mat))))

        frame = tf.reduce_sum(synt_scores, axis=1, keep_dims=False)

        output = synt_scores #(synt_scores + sem_scores) # / 2.0
        rt = ReturnTensor(output)
        rt.sigmoid_output = tf.sigmoid(output)
        rt.sem_rep = sem_embed_mat
        rt.synt_rep = synt_embed_mat
        rt.synt_scores = synt_scores
        rt.sem_scores = sem_scores
        rt.frame = frame
        rt.debug_embed_rep = debug_embed_rep
        return rt

# return tf.nn.softmax(tf.transpose(self.scoring(pred_mat, context_tensor)), dim=1) #tf.transpose(pred_mat)













    def complete_embeddings(self, sent, pred_id_vector): #, sem_arg_embeddings, synt_arg_embeddings, synt_output_dim):
        # If just semantic / sp representation:
        if self.flags.use_sp and not self.flags.use_context:
            return tf.gather(self.sem_arg_embeddings, sent)
        # If just use context
        if not self.flags.use_sp and self.flags.use_context:
            return self.context_embeddings(sent, pred_id_vector, self.synt_arg_embeddings, output_dim=self.flags.context_dims)
        # Or else use both:
        sem_embeds = tf.gather(sem_arg_embeddings, sent)
        synt_embeds = self.context_embeddings(sent, pred_id_vector, self.synt_arg_embeddings, output_dim=self.flags.context_dims)
        return tf.concat([sem_embeds, synt_embeds], axis=1)

    def pred_full_scoring(self, preds, pred_embeddings, rep_mat):
        # Reshape the context mat to remove the batch dim
        context_tensor = rep_mat  # tf.reshape(rep_mat, shape=[-1, tf.shape(rep_mat)[-1]])
#        return context_tensor
        # Shape [batch_size, num_tokens, pred_embed_dim]
        # actually ,remove batch for now
        pred_mat = tf.gather(pred_embeddings, preds)
#        return(pred_mat)
        pred_mat = tf.squeeze(pred_mat)
        return tf.transpose(self.scoring(pred_mat, context_tensor))
#        return tf.nn.softmax(tf.transpose(self.scoring(pred_mat, context_tensor)), dim=1) #tf.transpose(pred_mat)

        # Split the pred embeddings into individual pred tensors
#        score_tensor = tf.map_fn(lambda x: self.scoring(x, context_tensor), pred_mat)

        # Reshape score tensor for loss tensor
#        score_tensor = tf.transpose(tf.reshape(score_tensor, shape=[-1, tf.shape(score_tensor)[-1]]))
#        return score_tensor

    def scoring(self, pred_tensor, rep_tensor):
        pred_tensor = tf.reshape(pred_tensor, shape=[tf.shape(rep_tensor)[-1], -1])
        return tf.matmul(rep_tensor, pred_tensor)

    def score(self):
        rep = self.complete_embeddings(self.sent_placeholder,
                                       self.sem_arg_embeddings,
                                       self.synt_arg_embeddings,
                                       self.synt_output_dims)
        return self.pred_full_scoring(self.pred_placeholder, self.pred_embeddings, rep)

    def loss(self):
        prediction = self.score() #self.pred_full_scoring(self.pred_placeholder, self.pred_embeddings, rep_mat)
        loss = tf.reduce_sum(self.gold_placeholder - prediction)
#        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.transpose(prediction),
#                                                      labels=tf.transpose(self.gold_placeholder))
        return loss

    def loss_old(self, gold, preds, pred_embeddings, rep_mat):
        prediction = self.pred_full_scoring(preds, pred_embeddings, rep_mat)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.transpose(prediction),
                                                       labels=tf.transpose(gold))
        return loss

    def decode(self, sess, instance):
        sent_ids, pred_ids, gold_labels = separate_pred_dicts(instance)
        feed_dict={
            sent_placeholder: sent_ids,
            pred_placeholder: pred_ids,
            gold_placeholder: gold_labels
        }
        scored_mat_opt = self.score() # pred_full_scoring(self.pred_placeholder, self.pred_embeddings)
        sess.run(scored_mat_opt, feed_dict=feed_dict)

        pred_indices = [i for i, w in enumerate(instance.words) if w.is_prd]
        num_preds = len(pred_indices)
        scored_by_pred = np.split(scores, indices_or_sections=num_preds, axis=1)
        pases = []
        for pred_id, scored_mat in zip(pred_indices, scored_by_pred):
            #        print(pred_id)
            pred = Predicate(word_index=pred_id,
                             word_form=instance.words[pred_id].form)
            args = []
            for slot, slot_id in zip(arg_types, range(num_types)):
                if slot != 'NIL':
                    max_index = np.argmax(scored_mat[slot_id])
                    args.append(Argument(word_index=max_index,
                                         word_form=instance.words[max_index].form,
                                         arg_type=slot))
                    #                print('\t', max_index)
            pases.append(PAS(pred, args))
        return Sentence(instance.words, pases)


    def context_embeddings(self, sent, pred_id_vector, arg_embeddings, output_dim):
        # Shape [batch_size, max_sent_len, emb_dim]
        embed_rep = tf.gather(arg_embeddings, sent)
        embed_rep = tf.concat([embed_rep, tf.expand_dims(pred_id_vector,1)], 1)
        embed_rep = tf.expand_dims(embed_rep, 1)
        # List of length max_sent_len, comprising [batch_size, emb_dim] tensors
        fw_cell = tf.contrib.rnn.LSTMCell(num_units=output_dim / 2, state_is_tuple=True)
        bw_cell = tf.contrib.rnn.LSTMCell(num_units=output_dim / 2, state_is_tuple=True)

        # Construct RNN cells for all layers
        num_units = num_units=output_dim / 2
        dropout = self.flags.context_dropout
        fw_single_cells = []
        bw_single_cells = []

        for _ in range(self.flags.num_context_layers):
#            fw_single_cell =  tf.contrib.rnn.LayerNormBasicLSTMCell(num_units) #tf.contrib.rnn.GRUCell(num_units)  # Or LSTMCell(num_units)
            fw_single_cell =  tf.contrib.rnn.LSTMCell(num_units, state_is_tuple=True) #tf.contrib.rnn.GRUCell(num_units)  # Or LSTMCell(num_units)
            do_fw_single_cell = tf.contrib.rnn.DropoutWrapper(fw_single_cell, output_keep_prob=1.0 - dropout)
#            bw_single_cell =  tf.contrib.rnn.LayerNormBasicLSTMCell(num_units) # tf.contrib.rnn.GRUCell(num_units)  # Or LSTMCell(num_units)
            bw_single_cell =  tf.contrib.rnn.LSTMCell(num_units, state_is_tuple=True) # tf.contrib.rnn.GRUCell(num_units)  # Or LSTMCell(num_units)
            do_bw_single_cell = tf.contrib.rnn.DropoutWrapper(bw_single_cell, output_keep_prob=1.0 - dropout)
            fw_single_cells.append(do_fw_single_cell)
            bw_single_cells.append(do_bw_single_cell)
        # Add into multi cell
        fw_cell = tf.contrib.rnn.MultiRNNCell(fw_single_cells)
        bw_cell = tf.contrib.rnn.MultiRNNCell(bw_single_cells)

        #        return embed_rep
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell,
            cell_bw=bw_cell,
            dtype=tf.float64,
            #        sequence_length=X_lengths,
            inputs=embed_rep)
        output_fw, output_bw = outputs
        states_fw, states_bw = states
        final_rep = tf.squeeze(tf.concat(outputs, 2))
        return final_rep