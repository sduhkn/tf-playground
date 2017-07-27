# coding:utf-8
from __future__ import print_function, division
import os
import collections
import numpy as np
import tensorflow as tf

path = os.getcwd()
poetry_file = "data/poetry.txt"
file_path = os.path.join(path, poetry_file)

poetrys = list()
with open(file_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        title_content = line.split(":")
        if len(title_content) == 2:
            title, content = title_content
            if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                continue
            if len(content) < 5 or len(content) > 79:
                continue
            poetrys.append(content)
        else:
            continue

poetrys = sorted(poetrys, key=lambda poetry1: len(poetry1))
print('唐诗总数为：{}'.format(len(poetrys)))

all_words = list()
for poetry in poetrys:
    all_words.extend([word for word in poetry])
counter = collections.Counter(all_words)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])

words, _ = zip(*count_pairs)
print(words)
nb_word = len(words)
nb_poetry = len(poetrys)
word2num = dict(zip(words, range(nb_word)))
to_num = lambda word: word2num.get(word, nb_word)
poetrys_vector = [list(map(to_num, poetry)) for poetry in poetrys]

batch_size = 128
nb_batch = nb_poetry // batch_size
x_batches = list()
y_batches = list()
for i in range(nb_batch):
    start_index = i * batch_size
    end_index = start_index + batch_size

    batch_data = poetrys_vector[start_index:end_index]
    length = max(map(len, batch_data))
    xdata = np.full((batch_size, length), 0, np.int32)
    for row in range(batch_size):
        xdata[row, :len(batch_data[row])] = batch_data[row]
    ydata = np.copy(xdata)
    ydata[:, :-1] = xdata[:, 1:]

    x_batches.append(xdata)
    y_batches.append(ydata)

# [batch_size, time_step]
input_data = tf.placeholder(tf.int32, [batch_size, None])
output_data = tf.placeholder(tf.int32, [batch_size, None])


def neural_network(model='lstm', rnn_size=128, num_layers=2):
    cell_fun = None
    if model == "rnn":
        cell_fun = tf.nn.rnn_cell.BasicRNNCell
    elif model == "gru":
        cell_fun = tf.nn.rnn_cell.GRUCell
    elif model == "lstm":
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell

    cell = cell_fun(rnn_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    initial_state = cell.zero_state(batch_size, dtype=tf.float32)
    with tf.variable_scope('rnnlm'):
        softmax_w = tf.get_variable("softmax_w", shape=[rnn_size, nb_word + 1])
        softmax_b = tf.get_variable("softmax_b", shape=[nb_word + 1])
        with tf.device("/cpu:0"):
            embeding = tf.get_variable("embeding", shape=[nb_word + 1, rnn_size])
            # [batch_size, time_step, rnn_size]
            inputs = tf.nn.embedding_lookup(embeding, input_data)

    outputs, last_state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, initial_state=initial_state)
    output = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
    logits = tf.matmul(output, softmax_w) +softmax_b
    probs = tf.nn.softmax(logits)

    return logits, last_state, probs, cell, initial_state

def train():
    logits, last_state, probs, _, _ = neural_network(model='lstm', rnn_size=128, num_layers=2)
    targets = tf.reshape(output_data, [-1])
    loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets],
                                                  [tf.ones_like(targets, dtype=tf.float32)], nb_word)
    cost = tf.reduce_mean(loss)
    lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())
        for epoch in range(50):
            state = last_state
            sess.run(tf.assign(lr, 0.002 * (0.97 ** epoch)))
            n = 0
            for batch in range(nb_batch):
                feed_dict = {
                    input_data: x_batches[n],
                    output_data: y_batches[n]
                }
                train_loss, last_state, _ = sess.run([cost, last_state, train_op], feed_dict=feed_dict)
                n += 1
                print("迭代第 {} 次, 第 {} 批, loss is {}".format(epoch, batch, train_loss))
            if epoch % 10 == 0 and epoch != 0:
                saver.save(sess, "./save/model.ckpt", global_step=epoch)

if __name__ == "__main__":
    train()
