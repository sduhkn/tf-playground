from SentimentCNN import TextCNN, data_helper
import tensorflow as tf
import os
import pickle
"""
    train或者test模式通过修改
    tf.flags.DEFINE_string("train_or_test", "train", "train or test")
    中的第二个参数
"""

data_type = data_helper.Data_type
# Parameters
# ==================================================
tf.flags.DEFINE_string("train_or_test", "test", "Value can be selected by train or test")
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_size", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


#
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))

def train(config, vocab):
    assert isinstance(config, dict), "config is not dict type"
    start_id = 0    # 设置step初始的id
    graph = tf.Graph()
    with graph.as_default():
        # 分配显存
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement,
            gpu_options=gpu_options
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            with tf.variable_scope("model", reuse=False):
                model = TextCNN.TextCNN(**config)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            if os.path.exists("./save/checkpoint"):
                ckpt = tf.train.get_checkpoint_state("./save")
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 定位当前保存的global_step
                start_id = int(ckpt.model_checkpoint_path.split("-")[1]) + 1
            data_label = vocab.generate_data_label(data_type=data_type.TRAIN, pad_len=30)
            for step in range(start_id, 100):
                count = 0
                losses = 0
                accuracy = 0
                for i, (data, label) in enumerate(vocab.batch_iter(data_label)):
                    train_loss, train_accuracy = model.run_train_step(sess, data, label)
                    losses += train_loss
                    accuracy += train_accuracy
                    count += 1
                print("step: {}, loss:{:4f} accuracy: {:4f}".format(step,
                                                                    losses/count, accuracy/count))
                if step % 5 == 0 and step != 0:
                    checkpoint_path = os.path.join("./save", "model.ckpt")
                    saver.save(sess, checkpoint_path, global_step=step)


def eval(config, vocab):
    assert isinstance(config, dict)
    ckpt = tf.train.get_checkpoint_state("./save")
    graph = tf.Graph()
    with graph.as_default():
        # 分配显存
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement,
            gpu_options=gpu_options
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            with tf.variable_scope("model"):
                valid_model = TextCNN.TextCNN(**config, is_train=False)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            test_data_label = vocab.generate_data_label(data_type=data_type.TEST, pad_len=30)
            count = 0
            losses = 0
            accuracy = 0
            for i, (data, label) in enumerate(vocab.batch_iter(test_data_label)):
                test_loss, test_accuracy = valid_model.run_test_step(sess, data, label)
                losses += test_loss
                accuracy += test_accuracy
                count += 1
            print("test \t {} {}".format(losses/count, accuracy/count))


def main(_):
    cur_dir = os.getcwd()
    data_dir = os.path.join(cur_dir, "data")
    neg_file = "rt-polarity.neg"
    pos_file = "rt-polarity.pos"
    pos_path = os.path.join(data_dir, pos_file)
    neg_path = os.path.join(data_dir, neg_file)
    if os.path.exists("./vocab.pkl"):
        with open("./vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
    else:
        vocab = data_helper.Vocab(pos_path, neg_path)
        with open("./vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)
    config = {
        "seq_len": 30,
        "num_classes": 2,
        "vocab_size": vocab.count + 1,
        "embeding_size": FLAGS.embedding_size,
        "filter_sizes": list(map(int, FLAGS.filter_sizes.split(","))),
        "num_filters": FLAGS.num_filters,
        "l2_reg_lambda": FLAGS.l2_reg_lambda,
    }
    if FLAGS.train_or_test == "train":
        config["embedding_matrix"] = vocab.embedding
        train(config, vocab)
    elif FLAGS.train_or_test == "test":
        eval(config, vocab)
    else:
        raise ValueError("FLAGS.train_or_test is not right")

if __name__ == '__main__':
    tf.app.run()
