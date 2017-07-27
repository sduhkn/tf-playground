import xml.sax
import re
import jieba
stop_word = ["的", "得"]
punctuation = ["(", ")", " ", ":", ",", "“", "”"]
# Special tokens
# PARAGRAPH_START = '<p>'
# PARAGRAPH_END = '</p>'
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
# DOCUMENT_START = '<d>'
# DOCUMENT_END = '</d>'


class Vocab(object):
    """
        记录word和id的映射关系
    """
    def __init__(self, data):
        self._count = 0
        words_set = set()
        other_token = [PAD_TOKEN, SENTENCE_START, SENTENCE_START, UNKNOWN_TOKEN]
        other_len = len(other_token)
        for sentence in data:
            for word in sentence:
                if word not in punctuation:
                    words_set.add(word)
        self._count = len(words_set) + other_len
        # 注意这里是从3~count  要考虑到其他token
        self._word2id = dict(zip(words_set, range(other_len, self._count+other_len)))
        self._id2word = dict(zip(range(other_len, self._count+other_len), words_set))

        for i, w in enumerate(other_token):
            self._word2id.update({w: i})
            self._id2word.update({i: w})


        print("total word is {}".format(self._count))

    def IdToWord(self, word_id):
        return self._id2word.get(word_id, UNKNOWN_TOKEN)

    def WordToId(self, word: str):
        return self._word2id.get(word, 3)
    @property
    def count(self):
        return self._count


def Pad(ids, pad_id, length):
    """
    Pad or trim list to len length.
    Args:
    ids: list of ints to pad
    pad_id: what to pad with
    length: length to pad or trim to

    Returns:
    ids trimmed or padded with pad_id
    """
    assert pad_id is not None
    assert length is not None

    if len(ids) < length:
        a = [pad_id] * (length - len(ids))
        return ids + a
    else:
        return ids[:length]


def GetWordIds(text, vocab, pad_len=None, pad_id=None):
    """Get ids corresponding to words in text.
    将text的word转成id
    Assumes tokens separated by space.

    Args:
    text: a string
    vocab: TextVocabularyFile object
    pad_len: int, length to pad to
    pad_id: int, word id for pad symbol

    Returns:
    A list of ints representing word ids.
    """
    ids = []
    for w in text:
        i = vocab.WordToId(w)
        if i >= 0:
            ids.append(i)
        else:
            ids.append(vocab.WordToId(UNKNOWN_TOKEN))
    if pad_len is not None:
        return Pad(ids, pad_id, pad_len)
    return ids


def Ids2Words(ids_list, vocab):
    """Get words from ids.
    根据id转成相应的word
    Args:
    ids_list: list of int32
    vocab: TextVocabulary object

    Returns:
    List of words corresponding to ids.
    """
    # assert isinstance(ids_list, list), '%s  is not a list' % ids_list
    return [vocab.IdToWord(i) for i in ids_list]


def parse_data():
    contents = []
    titles = []
    # res = r'<a.*?>(.*?)</a>.*?<content>(.*?)</content>'
    # mm = re.findall(res, content1, re.S|re.M)
    # print(mm)
    # if not mm[0][1]:
    #     print("is nul")
    res = r'<contenttitle>(.*?)</contenttitle>.*?<content>(.*?)</content>'
    with open("./SogouCS.WWW08.txt", "rb") as f:
        raw_content = ""
        for line in f:
            if len(titles) > 10000:
                break
            if line.strip().startswith(b"<doc>"):
                continue
            elif line.strip().startswith(b"</doc>"):
                t = re.findall(res, raw_content, re.S | re.M)
                title, content = t[0]
                if content and title:
                    contents.append(content)
                    titles.append(title)
                raw_content = ""
            else:
                raw_content += line.decode("utf-8").replace(r"\u3000", "").replace("\n", "")
    return contents, titles


if __name__ == '__main__':
    contents, titles = parse_data()
    for content in contents:
        print(content)
    # s = "新华网武汉4月21日电(记者郑道锦 李鹏翔)在2:3输给中国女足(中国女足新闻,中国女足说吧)后," \
    #     "世界明星联队主教练之一的鲍威尔感慨地说,没想到中国队当晚能踢得这样好"
    # ss = [word for word in jieba.cut(s, cut_all=True) if word not in stop_word]
    # print(ss)
    vocab = Vocab(contents + titles)
    print(vocab._word2id)




