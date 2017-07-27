import numpy as np
import re
import os
import math
from enum import Enum
from gensim.models.word2vec import Word2Vec

stop_word = [",", "the", "a", "and", "of", "to", "'s", "is", "was", ""]


class Data_type(Enum):
    TRAIN = "train"
    TEST = "test"


class Vocab(object):
    def __init__(self, pos_path, neg_path, seed=0, train_test_split=0.8):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0
        self._seed = seed
        self.embedding = list()
        np.random.seed(self._seed)
        word_list = list()
        pos_data = []
        neg_data = []
        with open(pos_path, "r", encoding="utf-8") as f:
            for _, line in enumerate(f):
                sentence = clean_str(line.strip()).split()
                sentence = list(filter(lambda x: x not in stop_word, sentence))
                word_list.extend(sentence)
                pos_data.append(sentence)
        with open(neg_path, "r", encoding="utf-8") as f:
            for _, line in enumerate(f):
                sentence = clean_str(line.strip()).split()
                sentence = list(filter(lambda x: x not in stop_word, sentence))
                word_list.extend(sentence)
                neg_data.append(sentence)
        self.word_set = set(word_list)
        self._count = len(self.word_set)
        self._word_to_id = dict(zip(self.word_set, range(1, self._count + 1)))
        self._id_to_word = dict(zip(range(1, self._count + 1), self.word_set))
        # 从pos和neg data中各选出相应比例的数据加入进去  维持平衡
        train_pos_count = math.ceil(len(pos_data) * train_test_split)
        train_neg_count = math.ceil(len(neg_data) * train_test_split)
        pos_label = [[0, 1] for _ in pos_data]
        neg_label = [[1, 0] for _ in neg_data]
        self.train_data = pos_data[:train_pos_count] + neg_data[:train_neg_count]
        self.train_label = pos_label[:train_pos_count] + neg_label[:train_neg_count]

        self.test_data = pos_data[train_pos_count:] + neg_data[train_neg_count:]
        self.test_label = pos_label[train_pos_count:] + neg_label[train_neg_count:]
        np.random.shuffle(self.train_data)
        np.random.shuffle(self.train_label)
        np.random.shuffle(self.test_data)
        np.random.shuffle(self.test_label)
        self.w2v_model(pos_data + neg_data)
        print("word2vec模型加载完成")
        del pos_data, neg_data

    def w2v_model(self, sentense):
        model = Word2Vec(sentense, min_count=1)
        model.wv.save_word2vec_format("./w2v.model", binary=False)
        print("w2v.model保存成功")
        # 先根据 word_id 确定 word
        # 根据word查找embedding
        embedding = list()
        embedding.append([0] * 100)
        for i in range(1, self._count + 1):
            embedding.append(model.wv[self.id_to_word(i)])
        self.embedding = np.array(embedding)
        np.save("./embedding", self.embedding)
        del embedding

    def generate_data_label(self, data_type, pad_len=None):
        if not data_type:
            raise ValueError("not set data_type")
        if data_type == Data_type.TRAIN:
            data = self.train_data
            label = self.train_label
        elif data_type == Data_type.TEST:
            data = self.test_data
            label = self.test_label
        else:
            raise ValueError("the data_type is not right. plz choose train or test")
        assert len(data) == len(label), "the length of data and label is not equal"
        data = [GetWordIds(sentence, self.word_to_id, pad_len=pad_len) for sentence in data]
        return data, label

    def batch_iter(self, data_label, batch_size=32):
        data, label = data_label
        assert len(data) == len(label), "the length of data and label is not equal"
        data = np.array(data)
        data_size = len(data)
        num_batch_per_epoch = int((data_size - 1) / batch_size) + 1
        for batch_num in range(num_batch_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            print(end_index)
            yield data[start_index: end_index], label[start_index: end_index]

    def word_to_id(self, word):
        return self._word_to_id.get(word, 0)

    def id_to_word(self, word_id):
        return self._id_to_word.get(word_id, "UNKNOWN")

    @property
    def count(self):
        return self._count


def pad(ids, pad_len, pad_id):
    assert pad_id is not None
    assert pad_len is not None
    if len(ids) < pad_len:
        a = [pad_id] * (pad_len - len(ids))
        return ids + a
    else:
        return ids[:pad_len]


def GetWordIds(text, word_to_id, pad_len=None, pad_id=0):
    """Get ids corresponding to words in text.
      Assumes tokens separated by space.
      Args:
        text:  list of string
        word_to_id: dict
        pad_len: int, length to pad to
        pad_id: int, word id for pad symbol
      Returns:
        A list of ints representing word ids.
      """
    ids = []
    for word in text:
        i = word_to_id(word)
        ids.append(i)
    if pad_len is not None:
        return pad(ids, pad_len, pad_id)
    return ids


def Ids2Words(ids_list, vocab):
    """Get words from ids.

    Args:
      ids_list: list of int32
      vocab: TextVocabulary object

    Returns:
      List of words corresponding to ids.
    """
    assert isinstance(ids_list, list), '%s  is not a list' % ids_list
    return [vocab.id_to_word(i) for i in ids_list]


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
