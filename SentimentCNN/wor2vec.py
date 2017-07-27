import os
from gensim.models.word2vec import Word2Vec
from SentimentCNN.data_helper import clean_str, Vocab
import pickle
import numpy as np

def w2v_model():
    cur_dir = os.getcwd()
    data_dir = os.path.join(cur_dir, "data")
    neg_file = "rt-polarity.neg"
    pos_file = "rt-polarity.pos"
    pos_path = os.path.join(data_dir, pos_file)
    neg_path = os.path.join(data_dir, neg_file)
    stop_word = [",", "the", "a", "and", "of", "to", "'s", "is", "was", ""]
    data = list()
    with open(pos_path, "r", encoding="utf-8") as f:
        for _, line in enumerate(f):
            s = clean_str(line.strip()).split()
            s = list(filter(lambda x: x not in stop_word, s))
            data.append(s)
    with open(neg_path, "r", encoding="utf-8") as f:
        for line in f:
            s = clean_str(line.strip()).split()
            s = list(filter(lambda x: x not in stop_word, s))
            data.append(s)

    model = Word2Vec(data, min_count=1)
    print(model.wv["film"])
    model.wv.save_word2vec_format("./w2v.model", binary=False)

def read_w2v_model():
    with open("./vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    print(vocab.count)
    wid_vec_dict = dict()
    wid_vec_dict[0] = 100 * [0]
    with open("./w2v.model", "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            else:
                word, *data = line.split(" ")
                if word not in vocab.word_set:
                    raise ValueError("can not find the word")
                wid = vocab.word_to_id(word)
                wid_vec_dict[wid] = list(map(float, data))
    embedding = list()
    for i in range(0, vocab.count+1):
        embedding.append(wid_vec_dict[i])

    print(embedding[1])
    print(wid_vec_dict[1])
    print(len(embedding))

if __name__ == '__main__':
    read_w2v_model()
