import tensorflow as tf
import sys, os
import numpy as np
from SumModel.model import HParams
from SumModel.batch_reader import Batcher
from SumModel.utils import Vocab, parse_data, Ids2Words
from SumModel.model import SumModel


tf.flags.DEFINE_string("data_path", "./data/",
                       "Data Path")
tf.flags.DEFINE_string("mode", "test", "train or test")
tf.flags.DEFINE_integer('max_article_sentences', 50,
                        'Max number of first sentences to use from the '
                        'article')
tf.flags.DEFINE_integer('max_abstract_sentences', 20,
                        'Max number of first sentences to use from the '
                        'abstract')
tf.flags.DEFINE_string('save_dir', "./save",
                        'ckpt save directory')
FLAGS = tf.flags.FLAGS


def _Train(model: SumModel, batcher):
    start_id = 0
    with tf.Graph().as_default(), tf.Session() as sess:
        model.build_graph()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        if os.path.exists(os.path.join(FLAGS.save_dir, "checkpoint")):
            ckpt = tf.train.get_checkpoint_state(FLAGS.save_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
            # 定位当前保存的global_step
            start_id = int(ckpt.model_checkpoint_path.split("-")[1]) + 1
            print("继续上次开始训练")
        for step in range(start_id, 100):
            losses = 0
            count = 0
            for (article_batch, abstract_batch, targets,
                 article_lens, abstract_lens, loss_weights, _, _) in batcher.NextBatch():
                # print(article_batch)
                _, loss = model.run_train_step(sess, article_batch, abstract_batch, targets,
                                               article_lens, abstract_lens, loss_weights)
                losses += loss
                count += 1
            print("step: {}, losses: {}".format(step, losses / count))
            if step % 5 == 0:
                checkpoint_path = os.path.join(FLAGS.save_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)

def _Decode(model, batcher, hps, vocab):
    ckpt = tf.train.get_checkpoint_state(FLAGS.save_dir)
    with tf.Graph().as_default(), tf.Session() as sess:
        model.build_graph()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        (article_batch, abstract_batch, targets,
         article_lens, abstract_lens, loss_weights,
         origin_articles, origin_abstract) = next(batcher.NextBatch())

        enc_outputs, states = model.get_encode_state(sess, article_batch, article_lens)

        outputs = []
        latest_tokens = np.array([vocab.WordToId("<s>")] * len(article_batch))[:, np.newaxis]
        print(latest_tokens.shape)
        # 没完
        for i in range(hps.dec_timesteps):
            # latest_tokens: [batch, dec_timesteps]
            # return : output: [batch, 1] states:[batch, 2*cell_size]
            output, states = model.run_eval_step(sess, latest_tokens, enc_outputs, states)
            latest_tokens = output
            outputs.append(output)
        outputs = np.concatenate(outputs, axis=1)

        # 1. 先通过run_eval_step获取第一个解码的结果
        # 2. 将第一解码的结果变成第二次解码的输入
        # 3. 反复执行1、2两步，循环N次，知道出现SENTENCE_END = '</s>'停止
        for output in outputs:
            print(Ids2Words(output, vocab))

def main(_):
    contents, titles = parse_data()
    vocab = Vocab(contents + titles)
    print(vocab._word2id)
    batch_size = 4
    hps = HParams(
        mode=FLAGS.mode,  # train, eval, decode
        min_lr=0.01,  # min learning rate.
        lr=0.15,  # learning rate
        batch_size=batch_size,
        enc_layers=4,
        enc_timesteps=120,
        dec_timesteps=30,
        min_input_len=2,  # discard articles/summaries < than this
        num_hidden=128,  # for rnn cell
        emb_dim=128,  # If 0, don't use embedding
        max_grad_norm=2,
        num_softmax_samples=4096)  # If 0, no sampled softmax.
    batcher = Batcher(vocab, hps, contents, titles)
    if hps.mode == "train":
        model = SumModel(hps, vocab)
        _Train(model, batcher)
    elif hps.mode == "test":
        decode_hps = hps._replace(dec_timesteps=1)
        model = SumModel(decode_hps, vocab)
        _Decode(model, batcher, hps, vocab)


if __name__ == '__main__':
    tf.app.run()
    # a= np.array([2] * 10)[:, np.newaxis]
    # print(a.shape)
    # print(isinstance(a[0], list))