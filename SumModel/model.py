import tensorflow as tf
import numpy as np
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import attention_decoder, sequence_loss, rnn_decoder
from collections import namedtuple
LSTMCell = tf.nn.rnn_cell.LSTMCell
HParams = namedtuple('HParams',
                     'mode, min_lr, lr, batch_size, '
                     'enc_layers, enc_timesteps, dec_timesteps, '
                     'min_input_len, num_hidden, emb_dim, max_grad_norm, '
                     'num_softmax_samples')


def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
    """Get a loop_function that extracts the previous symbol and embeds it.

    Args:
      embedding: embedding tensor for symbols.
      output_projection: None or a pair (W, B). If provided, each fed previous
        output will first be multiplied by W and added B.
      update_embedding: Boolean; if False, the gradients will not propagate
        through the embeddings.

    Returns:
      A loop function.
    """

    def loop_function(prev, _):
        """function that feed previous model output rather than ground truth."""
        if output_projection is not None:
            prev = tf.nn.xw_plus_b(
                prev, output_projection[0], output_projection[1])
        prev_symbol = tf.argmax(prev, 1)
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = tf.stop_gradient(emb_prev)
        return emb_prev

    return loop_function


class SumModel(object):
    def __init__(self, hps: HParams, vocab):
        self._hps = hps
        self._vocab = vocab


    def _add_placeholders(self):
        hps = self._hps
        self._articles = tf.placeholder(tf.int32,
                                        [None, hps.enc_timesteps],
                                        name="articles")
        self._abstracts = tf.placeholder(tf.int32,
                                         [None, hps.dec_timesteps],
                                         name="abstracts")
        self._targets = tf.placeholder(tf.int32,
                                       [None, hps.dec_timesteps],
                                       name="targets")
        self._article_lens = tf.placeholder(tf.int32, [None],
                                            name="article_lens")
        self._abstract_lens = tf.placeholder(tf.int32, [None],
                                             name="article_lens")
        self._loss_weights = tf.placeholder(tf.float32,
                                            [None, hps.dec_timesteps],
                                            name='loss_weights')

    def run_train_step(self, sess: tf.Session, article_batch, abstract_batch, targets,
                       article_lens, abstract_lens, loss_weights):
        to_return = [self._train_op, self._loss]
        return sess.run(to_return,
                        feed_dict={self._articles: article_batch,
                                   self._abstracts: abstract_batch,
                                   self._targets: targets,
                                   self._article_lens: article_lens,
                                   self._abstract_lens: abstract_lens,
                                   self._loss_weights: loss_weights})

    def get_encode_state(self, sess: tf.Session, article_batch, article_lens):
        feed_dict = {self._articles: article_batch,
                     self._article_lens: article_lens}
        to_return = [self._enc_outputs, self._dec_in_state]
        return sess.run(to_return, feed_dict=feed_dict)

    def run_eval_step(self, sess, latest_tokens, enc_outputs, dec_init_states):
        feed_dict = {
            self._enc_outputs: enc_outputs,
            self._dec_in_state: dec_init_states,
            self._abstracts: latest_tokens,
            self._abstract_lens: np.ones([len(dec_init_states)], np.int32)}
        to_return = [self._outputs, self._dec_out_state]
        return sess.run(to_return, feed_dict=feed_dict)

    def _seq2seq(self):
        hps = self._hps
        vocab_size = self._vocab.count
        with tf.variable_scope("SumModel"):
            article_lens = self._article_lens
            # 由于sequence loss需要 seq_len * [batch_size]
            targets = tf.unstack(tf.transpose(self._targets))
            loss_weights = tf.unstack(tf.transpose(self._loss_weights))
            with tf.variable_scope('embedding'), tf.device('/cpu:0'):
                embedding = tf.get_variable("embedding", [vocab_size, hps.emb_dim],
                                            dtype=tf.float32)
                # [batch, seq_len, emb_dim]
                emb_encoder_inputs = tf.nn.embedding_lookup(embedding, self._articles)
                emb_decoder_inputs = tf.nn.embedding_lookup(embedding, self._abstracts)

            with tf.variable_scope("encoder"):
                cell_fw = LSTMCell(
                    hps.num_hidden,
                    initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                    state_is_tuple=False
                )
                cell_bw = LSTMCell(
                    hps.num_hidden,
                    initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
                    state_is_tuple=False
                )
                # outputs: (output_fw, output_bw) => output_fw: [batch_size, max_time, cell_fw.output_size]
                # output_states: A tuple (output_state_fw, output_state_bw)
                encoder_outputs, encoder_output_states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, inputs=emb_encoder_inputs, dtype=tf.float32,
                    sequence_length=article_lens)
            # encoder_outputs:　[batch_size, max_time, 2 * output_size]
            self._enc_outputs = tf.concat(encoder_outputs, axis=2)
            # [batch_size, 2 * output_size]
            encoder_state_fw, _ = encoder_output_states

            with tf.variable_scope("output_projection"):
                w = tf.get_variable("w", [hps.num_hidden, vocab_size], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=1e-4))
                v = tf.get_variable("b", [vocab_size], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=1e-4))

            with tf.variable_scope("decoder"):
                loop_function = None
                if hps.mode == "test":
                    loop_function = _extract_argmax_and_embed(embedding, (w, v),
                                                              update_embedding=False)
                decoder_cell = LSTMCell(
                    hps.num_hidden,
                    initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
                    state_is_tuple=False
                )
                # 将实际输入转化成符合要求的输入
                # [seq_len, batch, emb_dim] => seq_len * [batch, emb_dim]
                emb_decoder_inputs = tf.unstack(tf.transpose(emb_decoder_inputs, perm=[1, 0, 2]))
                # [batch, cell_size]
                self._dec_in_state = encoder_state_fw
                initial_state_attention = (hps.mode == 'test')
                # decoder_outputs: seq_len * [batch, hidden_size]
                # self._dec_out_state: [batch, state_size]=[batch, 2*cell_size]
                decoder_outputs, self._dec_out_state = attention_decoder(
                    decoder_inputs=emb_decoder_inputs, initial_state=self._dec_in_state,
                    attention_states=self._enc_outputs, cell=decoder_cell, num_heads=1,
                    loop_function=loop_function,
                    initial_state_attention=initial_state_attention
                )

            with tf.variable_scope("output"):
                # 还可以写成
                #[batch * seq_len, vsize]
                output = tf.reshape(tf.stack(values=decoder_outputs, axis=1), [-1, hps.num_hidden])
                logits = tf.matmul(output, w) + v
                model_outputs = tf.unstack(tf.reshape(logits, [-1, hps.dec_timesteps, vocab_size]),
                                           axis=1)
                # seq_len * [batch, vsize]
                # 输出层共享
                # model_outputs = []
                # for i in range(len(decoder_outputs)):
                #     if i > 0:
                #         tf.get_variable_scope().reuse_variables()
                #     model_outputs.append(
                #         tf.nn.xw_plus_b(decoder_outputs[i], w, v))

            with tf.variable_scope("loss"):
                # logits: seq_len * [batch_size, vsize]
                # targets: seq_len * [batch_size]
                # weights: seq_len * [batch_size] 注意这里的weights的作用是做mask
                # 1. sequence_loss先是调用sequence_loss_by_example,获取[batch_size]维的loss，在除以batch_size
                # 2. sequence_loss_by_example利用weights来做mask，获取实际的每个time_step的平均loss
                # 因为batch里面实际句子长度不一样，所有weights要先初始化zeros,然后向里面填1
                self._loss = sequence_loss(logits=model_outputs, targets=targets,
                                           weights=loss_weights)
            if hps.mode == "test":
                with tf.variable_scope("decode_output"):
                    # seq_len * [batch, vsize] => seq_len * [batch, 1]
                    best_outputs = [tf.arg_max(x, 1) for x in model_outputs]
                    # [batch, seq_len]
                    self._outputs = tf.concat(
                        axis=1, values=[tf.reshape(x, [-1, 1]) for x in best_outputs])
                    # self._topk_log_probs, self._topk_ids = tf.nn.top_k(
                    #     tf.log(tf.nn.softmax(model_outputs[-1])), 5 * 2)

    def _add_train_op(self):
        hps = self._hps

        self._lr_rate = 0.15
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self._loss, tvars), hps.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr_rate)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars), global_step=self.global_step, name="train_step")

    def assign_global_step(self, sess:tf.Session, new_value: int):
        sess.run(tf.assign(self.global_step, new_value))

    def build_graph(self):
        self._add_placeholders()
        self._seq2seq()
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        if self._hps.mode == "train":
            self._add_train_op()
