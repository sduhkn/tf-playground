from collections import namedtuple
from random import shuffle
import numpy as np
import tensorflow as tf
from SumModel.utils import Vocab
from SumModel import utils
from SumModel.model import HParams
from queue import deque

ModelInput = namedtuple('ModelInput',
                        'enc_input dec_input target enc_len dec_len '
                        'origin_article origin_abstract')


class Batcher(object):
    def __init__(self, vocab: Vocab, hps: HParams, contents, titles):
        self._vocab = vocab
        self._hps = hps
        self._contents = contents
        self._titles = titles
        self._fillInput()
        self._fillBucket()


    def _fillInput(self):
        hps = self._hps
        contents = self._contents
        titles = self._titles
        self._inputs = []

        start = utils.SENTENCE_START
        end_id = self._vocab.WordToId(utils.SENTENCE_END)
        pad_id = self._vocab.WordToId(utils.PAD_TOKEN)

        for i, article in enumerate(contents):
            abstract = titles[i]
            enc_input = []
            dec_input = [start]
            # enc_len = min(len(article), self._max_article_sentences)
            # enc_input.extend(article[:enc_len])
            # dec_len = min(len(abstract), self._max_abstract_sentences)
            # dec_input.extend(abstract[:dec_len]) #utils.GetWordIds(abstract[j], self._vocab)
            if (len(article) < hps.min_input_len or
                        len(abstract) < hps.min_input_len):
                tf.logging.warning("Drop an example - too short")
                continue
            # 将多余的截断
            if len(article) > hps.enc_timesteps:
                enc_input = [article[i] for i in range(hps.enc_timesteps)]
            else:
                enc_input = [article[i] for i in range(len(article))]
            if len(abstract) > hps.dec_timesteps:
                dec_input.extend([abstract[i] for i in range(hps.dec_timesteps)])
            else:
                dec_input.extend([abstract[i] for i in range(len(abstract))])
            # 记录实际长度
            enc_input_len = len(enc_input)
            dec_output_len = len(dec_input)
            # pad and 汉字转成int
            enc_input = utils.GetWordIds(enc_input, self._vocab,
                                         hps.enc_timesteps, pad_id)
            dec_input = utils.GetWordIds(dec_input, self._vocab,
                                         hps.dec_timesteps, pad_id)
            target = dec_input[1:]
            target.append(end_id)

            element = ModelInput(enc_input, dec_input, target,
                                 enc_input_len, dec_output_len,
                                 article, abstract)
            self._inputs.append(element)

    def _fillBucket(self):
        hps = self._hps
        inputs = self._inputs
        inputs = sorted(inputs, key=lambda inp: inp.enc_len)
        self._batches = []
        for i in range(0, len(inputs), hps.batch_size):
            self._batches.append(inputs[i: i + hps.batch_size])

    def NextBatch(self):
        hps = self._hps
        for bucket in self._batches:
            batch_size = len(bucket)
            enc_batch = np.zeros(
                (batch_size, hps.enc_timesteps), dtype=np.int32)
            enc_input_lens = np.zeros(
                batch_size, dtype=np.int32)
            dec_batch = np.zeros(
                (batch_size, hps.dec_timesteps), dtype=np.int32)
            dec_output_lens = np.zeros(
                batch_size, dtype=np.int32)
            target_batch = np.zeros(
                (batch_size, hps.dec_timesteps), dtype=np.int32)
            loss_weights = np.zeros(
                (batch_size, hps.dec_timesteps), dtype=np.float32)
            origin_articles = ['None'] * batch_size
            origin_abstracts = ['None'] * batch_size
            for i in range(len(bucket)):
                (enc_inputs, dec_inputs, targets, enc_input_len, dec_output_len,
                 article, abstract) = bucket[i]

                origin_articles[i] = article
                origin_abstracts[i] = abstract
                enc_input_lens[i] = enc_input_len
                dec_output_lens[i] = dec_output_len
                enc_batch[i, :] = enc_inputs[:]
                dec_batch[i, :] = dec_inputs[:]
                target_batch[i, :] = targets[:]
                for j in range(dec_output_len):
                    loss_weights[i][j] = 1.0

            yield (enc_batch, dec_batch, target_batch, enc_input_lens, dec_output_lens,
                   loss_weights, origin_articles, origin_abstracts)


if __name__ == '__main__':
    a= "123"
    print(a.split())
