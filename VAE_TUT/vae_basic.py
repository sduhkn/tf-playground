from __future__ import division, print_function, absolute_import

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
current_dir = "../data/"
mnist = input_data.read_data_sets(current_dir, one_hot=True)

def img_show(image):
    image = image[1].reshape([28, 28])
    plt.imshow(image)
    plt.show()

class Layer:
    def __init__(self, input_data, n_output):
        self.input_data = input_data
        W = tf.Variable(tf.truncated_normal([int(self.input_data.get_shape()[1]), n_output], stddev=0.001),
                        name="W")
        b = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[n_output]), name="b")

        self.raw_output = tf.matmul(input_data, W) + b
        self.output = tf.nn.relu(self.raw_output)

n_X = 28*28
n_z = 20
with tf.name_scope("Input"):
    X = tf.placeholder(tf.float32, shape=[None, n_X], name="X_input")

# Encoder
with tf.name_scope("Encoder"):
    ENCODER_HIDDEN_COUNT = 400
    h1 = Layer(X, ENCODER_HIDDEN_COUNT).output
    # \mu(x)
    mu = Layer(h1, n_z).raw_output
    # \Sigma(x)
    log_sigma = Layer(h1, n_z).raw_output
    sigma = tf.exp(log_sigma)
    # epsilon = N(0, I)
    epsilon = tf.random_normal(tf.shape(sigma), name='eplison')
    # sample z
    z = mu + tf.exp(0.5 * log_sigma) * epsilon

# Decoder
with tf.name_scope("Decoder"):
    DECODER_HIDDEN_COUNT = 400
    layer1 = Layer(z, DECODER_HIDDEN_COUNT).output
    X_hat = Layer(layer1, n_X).raw_output

with tf.name_scope("Loss"):
    KL_distance = 0.5 * tf.reduce_sum(sigma + tf.pow(mu, 2) - log_sigma - 1,
                                      reduction_indices=1)
    decoder_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(X_hat, X),
                                 reduction_indices=1)
    loss = tf.reduce_mean(decoder_loss + KL_distance)
    loss_summary = tf.summary.scalar("loss", loss)

n_steps = 10000
lr = 0.01
batch_size = 32
train_op = tf.train.AdamOptimizer(lr).minimize(loss)
summary_op = tf.summary.merge_all()
fig = plt.figure()
ax = fig.add_subplot(211)
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter("./log/",
                                           graph=sess.graph)
    sess.run(tf.global_variables_initializer())
    for step in range(1, n_steps):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, l, summary_str = sess.run([train_op, loss, summary_op],
                                     feed_dict={X: batch_x})
        summary_writer.add_summary(summary_str, step)
        if step % 100 == 0:
            img1 = sess.run(X_hat, feed_dict={X: batch_x})[1].reshape([28, 28])
            # plt.imshow(img1)
            # plt.pause(1)
            print('Step:{0}, Loss:{1}'.format(step, l))


