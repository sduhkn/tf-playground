import jieba
import os
import codecs
from collections import Counter, defaultdict
import math

word2id = dict()
id2word = dict()

def read_data(path):
    for root, sub_dirs, files in os.walk(path):
        files_path = [os.path.join(root, file_path) for file_path in files]
    words = []
    for i, file_path in enumerate(files_path):
        doc = []
        with codecs.open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and line != '\\ufeff': [doc.extend(list(jieba.cut(sentence))) for sentence in line.split("/")]
        words.append(doc)
        # print("the sentence lengh is {0}, and the content is \n {1}".format(len(words), words))
    print(words)
    return words


def process(words):
    global word2id, id2word
    words_flatten = [word for doc in words for word in doc]
    words_set = Counter(words_flatten).keys()
    word2id = dict(zip(words_set, range(len(words_set))))
    id2word = dict(zip(range(len(words_set)), words_set))
    tmp_words = []
    for doc in words:
        tmp_doc = []
        for word in doc:
            tmp_doc.append(word2id.get(word))
        tmp_words.append(tmp_doc)
    return tmp_words
    # with open(path) as f:
    #     for line in f:
    #         print(line)


def IDF(docs, contain_doc):
    return math.log((docs / (contain_doc + 1)), 2)


def get_max_key_from_dict(data, k):
    data = sorted(data.items(), key=lambda x: x[1], reverse=True)
    return [data[i][0] for i in range(k)]


def tf_idf(words):
    global id2word
    print(id2word)
    docs_len = len(words)
    for doc in words:
        tf_count = Counter(doc)
        max_count = max(tf_count.values())
        tf_count = {key: value / max_count for key, value in tf_count.items()}
        contain_doc = 0
        for key in tf_count.keys():
            for doc in words:
                if key in doc:
                    contain_doc += 1
            tf_count[key] *= IDF(docs_len, contain_doc)
        topk_keys = get_max_key_from_dict(tf_count, 15)
        print([id2word.get(key, 0) for key in topk_keys])


if __name__ == '__main__':
    data = process(read_data("../data/专利"))
    tf_idf(data)
