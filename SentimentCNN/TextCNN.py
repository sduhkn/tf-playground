import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer


class TextCNN(object):
    def __init__(self,
                 seq_len, num_classes, vocab_size, embeding_size,
                 filter_sizes, num_filters, l2_reg_lambda=0.0, embedding_matrix=None,
                 is_train=True):
        """

        :param seq_len: 文本长度
        :param num_classes: 分类类别
        :param vocab_size:  单词个数
        :param embeding_size:
        :param filter_sizes: list
        :param num_filters: 卷积层中out_channel，卷积核个数
        :param l2_reg_lambda:
        """
        self.input_x = tf.placeholder(tf.int32, [None, seq_len], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)
        with tf.device("/cpu:0"), tf.variable_scope("embeding"):
            if embedding_matrix is not None:
                embedding = tf.Variable(embedding_matrix, name="embed", dtype=tf.float32)
            else:
                embedding = tf.get_variable("embed", shape=[vocab_size, embeding_size], dtype=tf.float32,
                                            initializer=tf.orthogonal_initializer())
                # embedding = tf.Variable(
                #     tf.random_uniform([vocab_size, embeding_size], -1.0, 1.0),
                #     name="embed")
            # [batch, seq, embed_size]
            self.embedded_chars = tf.nn.embedding_lookup(embedding, self.input_x)
            # [batch, seq, embed_size, 1] => [batch, width, height, in_channel]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        # len(filter_sizes) * [batch, 1, 1, num_filters]
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-{}".format(filter_size)):
                filter_shape = [filter_size, embeding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                                name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                # [batch, seq_len- filter_size+1, 1, num_filters]
                conv = tf.nn.conv2d(self.embedded_chars_expanded,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")

                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # [batch, 1, 1, num_filters]
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, seq_len - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding="VALID",
                                        name="pool")
                pooled_outputs.append(pooled)
        # flatten
        num_filters_total = num_filters * len(filter_sizes)
        # [batch, 1, 1, num_filters_total]
        self.h_pool = tf.concat(pooled_outputs, 3)
        # [batch, num_filters_total]
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.h_pool_flat = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.variable_scope("output"):
            W = tf.get_variable("W",
                                shape=[num_filters_total, num_classes],
                                initializer=xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            # [batch, num_classes]
            self.scores = tf.nn.xw_plus_b(self.h_pool_flat, W, b, name="scores")
            # [batch, 1]
            self.pred = tf.argmax(self.scores, 1, name="pred")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores,
                labels=self.input_y
            )
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(self.pred, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        if not is_train:
            return

        self._global_step = tf.Variable(1, name="global_step", trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                          10)
        optimizer = tf.train.AdamOptimizer(1e-3)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                  global_step=self._global_step)

    def assign_global_step(self, sess, global_step):
        sess.run(tf.assign(self._global_step, global_step))

    def run_train_step(self, sess, data, label):
        to_return = [self.train_op, self.loss, self.accuracy]
        feed_dict = {
            self.input_x: data,
            self.input_y: label,
            self.dropout_keep_prob: 0.5
        }
        _, loss, accuracy = sess.run(to_return, feed_dict=feed_dict)
        return loss, accuracy

    def run_test_step(self, sess, data, label):
        to_return = [self.loss, self.accuracy]
        feed_dict = {
            self.input_x: data,
            self.input_y: label,
            self.dropout_keep_prob: 1.0
        }
        loss, accuracy = sess.run(to_return, feed_dict=feed_dict)
        return loss, accuracy
