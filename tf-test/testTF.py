import tensorflow as tf
from tensorflow.python.util import nest
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.contrib.rnn import static_bidirectional_rnn, static_rnn
from tensorflow.contrib.legacy_seq2seq import attention_decoder

a = tf.ones([5, 4, 3], dtype=tf.float32) + \
    tf.Variable(tf.truncated_normal([5, 4, 3], mean=0, dtype=tf.float32), dtype=tf.float32)
a = tf.reshape(a, [-1])
t = tf.range(0, 4 * 5) * 3 + 1
r = tf.gather(a, t)
ran = tf.random_normal([3, 4, 2, 1], dtype=tf.float32)
ran_max = tf.reduce_max(ran, axis=1)


# aa = tf.sign(tf.ones([5,10]))
# a1 = tf.reduce_sum(aa, reduction_indices=1)
# test = tf.Variable(tf.random_uniform([4, 3], 0., 2.), dtype=tf.float32)
# test1 = tf.Variable(tf.random_uniform([4, 3], 0., 2.), dtype=tf.float32)
# test2 = tf.stack([test, test1], axis=1)
# test2_shape = tf.shape(test2)
def test_nest(sess):
    a = tf.random_normal([3, 4, 2], dtype=tf.float32)
    a_ = a.get_shape()
    b = nest.flatten(a)

    print(a_.with_rank_at_least(1))
    print(sess.run(b))


def test_global_key(sess):
    with tf.variable_scope("global"):
        f = tf.placeholder(tf.int32, shape=[2, 2])
        a = tf.Variable([1, 2, 3], name="a")
        b = tf.get_variable("b", shape=[1, 2])
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="global")
    sess.run(tf.global_variables_initializer())
    print(params)


def test_nameScope_values(sess, values):
    with tf.name_scope("test_name_scope", values=[values]):
        value = tf.convert_to_tensor(values, name="a")
        v = tf.constant(6, name="v")
        print(value.name)
        ad = tf.add_n([tf.constant([1, 2]), tf.constant([3, 4])])
        a = tf.constant([1, 2, 3], dtype=tf.int32)
        b = tf.constant([1, 0, 1], dtype=tf.int32)
        print(sess.run(a * b))


def add_variable(x):
    with tf.variable_scope("va"):
        w = tf.get_variable("w", shape=[2, 2], initializer=tf.truncated_normal_initializer(stddev=1e-4))
        b = tf.get_variable("b", shape=[2])
        weights = tf.convert_to_tensor(w, name="weights")
        biases = tf.convert_to_tensor(b, name="biases")
        mm = tf.matmul(x, weights)
        print(w.name)
    return mm


def test_for_loop(sess: tf.Session):
    x = tf.random_uniform(shape=[3, 2], minval=1, maxval=5)
    c = []
    with tf.variable_scope("out"):
        for i in range(5):
            if i > 0: tf.get_variable_scope().reuse_variables()
            c.append(add_variable(x))
    with tf.variable_scope("ooo"):
        for i in range(5):
            if i > 0: tf.get_variable_scope().reuse_variables()
            c.append(add_variable(x))
    sess.run(tf.global_variables_initializer())


def test_argmax(sess: tf.Session):
    a = tf.Variable(tf.random_uniform([4, 6, 7], minval=1, maxval=10))
    a1 = tf.constant(1, shape=[4, 6])
    ar = tf.arg_max(a, dimension=1)
    # print(a.get_shape().with_rank(1))
    print(a.get_shape().with_rank(2))


def test_assign_add(sess: tf.Session):
    x = tf.Variable(0.0)
    x_plus_1 = tf.assign_add(x, 1)

    with tf.control_dependencies([x_plus_1]):
        y = x

    print(sess.run(tf.global_variables_initializer()))
    for i in range(5):
        print(sess.run(y))


def test_squeeze(sess: tf.Session):
    a = tf.random_uniform([3, 1], minval=0, maxval=5, dtype=tf.int32)
    b = tf.squeeze(a)
    print(b.shape)


def test_gather(sess: tf.Session):
    a = tf.Variable(tf.random_uniform([3, 2], minval=1, maxval=5, dtype=tf.int32))
    b = tf.gather(a, indices=[[1, 2]])
    mask = tf.Variable(tf.random_uniform([3, 2], minval=0, maxval=2, dtype=tf.int32))
    c = a * mask
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    print(sess.run(mask))
    print(sess.run(tf.reshape(c, [-1])))


def test_pre(sess: tf.Session):
    a = tf.Variable([[1, 2, 3, 4, 5, 0, 0, 0], [1, 2, 3, 4, 0, 0, 0, 0]], dtype=tf.int32)
    b = tf.constant([5, 6], dtype=tf.int32)
    c = a[:, [3, 4]]
    sess.run(tf.global_variables_initializer())
    print(sess.run(c))


def test_topk(sess: tf.Session):
    a = tf.Variable(tf.random_uniform([4, 3], minval=0, maxval=6, dtype=tf.int32))
    b = tf.nn.top_k(a, k=2)
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    print(sess.run(tf.shape(b)))
    print(sess.run(b))


def test_scan(sess: tf.Session):
    elems = np.array([1, 0, 0, 0, 0, 0])
    initializer = tf.Variable([0, 1], dtype=tf.int32)
    fibonaccis = tf.scan(lambda a, _: tf.stack([a[1], a[0] + a[1]]), elems, initializer)
    sess.run(tf.global_variables_initializer())
    print(sess.run(fibonaccis))
    b = tf.stack([[1, 2, 10], [3, 4, 5]], axis=0)
    print(sess.run(b))
    print(sess.run(tf.shape(b)))


def test_feed(sess: tf.Session):
    a = tf.constant([[1, 2], [2, 3]])
    b = tf.reshape(a, [-1])
    rst = sess.run(b, feed_dict={a: [[2, 3], [4, 4]]})
    print(rst)


def test_while_loop(sess: tf.Session):
    i = tf.constant(0)
    c = lambda i: tf.less(i, 10)
    b = lambda i: tf.add(i, 1)
    r = tf.while_loop(c, b, [i])
    print(sess.run(r))


def test_tensorArray(sess: tf.Session):
    b = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.int32)
    a = tf.TensorArray(tf.int32, size=3)
    c = a.unstack(b)
    # for i in range(10):
    #     a = a.write(i, tf.constant([i, i+1], dtype=tf.int32))
    sess.run(tf.global_variables_initializer())
    print(sess.run(c.gather([1, 2])))


def main(_):
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()

        # sess.run(tf.global_variables_initializer())
        # print(sess.run(aa))
        # print(sess.run(test2_shape[1]))
        # test_nameScope_values(sess, tf.constant(1.0, shape=[1, 2], dtype=tf.float32, name="out_cons"))
        test_tensorArray(sess)
        sess.close()


if __name__ == '__main__':
    tf.app.run()
    # a = [1, 0, 2, 4]
    # b = list(filter(lambda x: x!=0, a))
    # print(b)
