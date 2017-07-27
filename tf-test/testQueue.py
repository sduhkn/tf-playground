import numpy as np
import tensorflow as tf
import time

NUM_THREADS = 2
N_SAMPLES = 500

x = np.random.randn(N_SAMPLES, 4) + 1         # shape (5, 4)
y = np.random.randint(0, 2, size=N_SAMPLES)   # shape (5, )
x2 = np.zeros((N_SAMPLES, 4))
print(y.shape)

# Define a FIFOQueue which each queue entry has 2 elements of length 4 and 1 respectively
queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.int32], shapes=[[4], []])

# Create an enqueue op to enqueue the [x, y]
enqueue_op1 = queue.enqueue_many([x, y])
enqueue_op2 = queue.enqueue_many([x2, y])

# Create an dequeue op
data_sample, label_sample = queue.dequeue()

# QueueRunner: create a number of threads to enqueue tensors in the queue.
# qr = tf.train.QueueRunner(queue, [enqueue_op1] * NUM_THREADS)
qr = tf.train.QueueRunner(queue, [enqueue_op1, enqueue_op2])

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    # Launch the queue runner threads.
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
    count = 0
    try:
        for step in range(1):
            count += 1
            if coord.should_stop():
                break
            one_data, one_label = sess.run([data_sample, label_sample])
            print("x = {} y = {}".format(one_data, one_label))
    except Exception as e:
        coord.request_stop(e)
    finally:
        coord.request_stop()
        coord.join(enqueue_threads)
