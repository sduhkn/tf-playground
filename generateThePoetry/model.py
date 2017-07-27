import tensorflow as tf
from tensorflow.python.ops import seq2seq
import numpy as np


class Model(object):
    def __init__(self, model='lstm', rnn_size=128, num_layers=2, batch_size=32, nb_word=None):
        # [batch_size, time_step]
        self.input_data = tf.placeholder(tf.int32, [batch_size, None])
        self.output_data = tf.placeholder(tf.int32, [batch_size, None])
        cell_fun = None
        if model == "rnn":
            cell_fun = tf.nn.rnn_cell.BasicRNNCell
        elif model == "gru":
            cell_fun = tf.nn.rnn_cell.GRUCell
        elif model == "lstm":
            cell_fun = tf.nn.rnn_cell.BasicLSTMCell

        cell = cell_fun(rnn_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

        self.initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", shape=[rnn_size, nb_word + 1])
            softmax_b = tf.get_variable("softmax_b", shape=[nb_word + 1])
            with tf.device("/cpu:0"):
                embeding = tf.get_variable("embeding", shape=[nb_word + 1, rnn_size])
                # [batch_size, time_step, rnn_size]
                inputs = tf.nn.embedding_lookup(embeding, self.input_data)

        outputs, last_state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, initial_state=self.initial_state)
        output = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)

        targets = tf.reshape(self.output_data, [-1])
        loss = tf.nn.seq2seq.sequence_loss_by_example([self.logits], [targets],
                                                      [tf.ones_like(targets, dtype=tf.float32)], nb_word)
        self.cost = tf.reduce_mean(loss)
        self.lr = tf.Variable(0.0, trainable=False)
        self.final_state = last_state
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
