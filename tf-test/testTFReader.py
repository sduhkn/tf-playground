import tensorflow as tf
from tensorflow.contrib.data import TextLineDataset, Iterator
# different from the testTF.py, use new API.

def traditional_read():
    filenames = ['./data.csv']
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    example, label = tf.decode_csv(value, record_defaults=[[1], [1]], field_delim=",")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for _ in range(10):
            print(sess.run(example))
        coord.request_stop()
        coord.join(threads)


def read_row(csv_row):
    record_defaults = [[0], [0], [0]]
    row = tf.decode_csv(csv_row, record_defaults=record_defaults)
    return row[:-1], row[-1]


def input_pipeline(filenames, batch_size) -> Iterator:
    dataset = TextLineDataset(filenames) \
        .skip(1) \
        .map(lambda line: read_row(line)) \
        .zip() \
        .shuffle(buffer_size=10) \
        .batch(batch_size)
    return dataset.make_initializable_iterator()


def new_read():
    iterator = input_pipeline(['data.csv'], 5)
    features, labels = iterator.get_next()

    nof_examples = 10
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        sess.run(iterator.initializer)
        while nof_examples > 0:
            nof_examples -= 1
            try:
                data_features, data_labels = sess.run([features, labels])
                print(data_features)
            except tf.errors.OutOfRangeError:
                pass


if __name__ == '__main__':
    new_read()
