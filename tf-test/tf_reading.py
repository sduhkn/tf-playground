import tensorflow as tf


def readMyFileFormat(fileNameQueue):
    reader = tf.TextLineReader()
    key, value = reader.read(fileNameQueue)

    record_defaults = [['null'], ['null'], [1]]
    e1, e2, label = tf.decode_csv(value, record_defaults=record_defaults, field_delim=",")
    example = tf.stack([e1, e2])
    return example, label


def inputPipeLine():
    # 单个Reader，单个样本
    filenames = ['./data/A.csv', './data/B.txt']
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=10, shuffle=False)
    example, label = readMyFileFormat(filename_queue)

    batchSize = 5
    min_after_dequeue = 8
    capacity = min_after_dequeue + 3 * batchSize
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batchSize, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch


def one_reader():
    example_batch, label_batch = inputPipeLine()
    sess = tf.Session()
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    sess.run(init)
    step = 0
    try:
        while not coord.should_stop():
            for i in range(10):
                e1, l1 = sess.run([example_batch, label_batch])
                print(e1, l1)
                step += 1
    except tf.errors.OutOfRangeError:
        print('step:{}'.format(step))
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    one_reader()
