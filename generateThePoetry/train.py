from __future__ import print_function, division
import os
import collections
import numpy as np
import tensorflow as tf
from model import Model


def train():
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

    batch_size = 32
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

        rnn_model = Model(nb_word=nb_word, batch_size=batch_size)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(tf.global_variables())
            for epoch in range(50):
                sess.run(tf.assign(rnn_model.lr, 0.002 * (0.97 ** epoch)))
                state = sess.run(rnn_model.initial_state)
                n = 0
                for batch in range(nb_batch):
                    feed_dict = {
                        rnn_model.input_data: x_batches[n],
                        rnn_model.output_data: y_batches[n],
                        rnn_model.initial_state: state
                    }
                    train_loss, state, _ = sess.run([rnn_model.cost, rnn_model.final_state, rnn_model.train_op],
                                                    feed_dict=feed_dict)
                    n += 1
                    print("迭代第 {} 次, 第 {} 批, loss is {}".format(epoch, batch, train_loss))
                if epoch % 10 == 0 and epoch != 0:
                    saver.save(sess, "./save/model.ckpt", global_step=epoch)

if __name__ == "__main__":
    train()