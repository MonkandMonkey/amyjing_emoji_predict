"""Utils for loading data, evaluating model."""
import re
import os
import json
import numpy as np
from collections import Counter

import keras
import keras.backend as K
import keras.datasets.imdb as imdb
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer


def filter_out_words(words, max_word_idx=None):
    """Filter out words whose idx is >= some idx.

    Keep all words if max_word_idx is None.

    Args:
        words: original word list.
        max_word_idx: word idx upper bound.

    Returns:
        List of words after filtering.
    """
    if max_word_idx is None:
        return words
    else:
        return [w for w in words if w < max_word_idx]


def load_data(train_file, test_file, max_word_idx=None, train_size=None):
    """Load train and test dataset from json file with specified vocab and train set size.

    Args:
        train_file: train data file with json format [{"text": [w1,w2,...wn], "label": 10}, ...].
        test_file: test data file with same json format as train_file.
        max_word_idx: word idx upper bound.
        train_size: train data size will be loaded.

    Returns:
        x_train, y_train, x_test, y_test
        x: numpy arrary [[23, 15, 67, 12, 89, 1], ...]
        y: numpy arrary [3,5,1,0,19, ...]
    """
    x_train, y_train, x_test, y_test = [], [], [], []

    with open(train_file, "r") as fr:
        train_data = json.load(fr)

    with open(test_file, "r") as fr:
        test_data = json.load(fr)

    if train_size is not None and train_size > 0:
        train_data = train_data[:train_size]

    for d in train_data:
        x_train.append(filter_out_words(d["text"], max_word_idx))
        cls = int(d["label"])
        y_train.append(cls)

    for d in test_data:
        x_test.append(filter_out_words(d["text"], max_word_idx))
        cls = int(d["label"])
        y_test.append(cls)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)


def train_tokenizer(texts, num_words=None):
    """Change word list to index list.

    Args:
        texts: text list, corpus to train the tokenizer.
        num_words: max word idx to keep (by frequency).

    Returns:
        tok: the trained tokenizer.
    """
    tok = Tokenizer(num_words=num_words)
    tok.fit_on_texts(texts)
    return tok


def texts_to_indexes(texts, tokenizer, num_words=None):
    """Change word list to index list.
        num_words: max word idx to keep (by frequency)

    Returns:
        train index list
        test index list
    """
    if tokenizer is None:
        tokenizer = train_tokenizer(texts, num_words=num_words)

    idx_list = tokenizer.texts_to_sequences(texts)
    return idx_list


def load_imdb_data():
    (x_train, y_train), (x_test, y_test) = imdb.load_data()
    print('x_train shape: {}'.format(x_train.shape))
    print('x_test shape: {}'.format(x_test.shape))
    print("x_train: {}".format(x_train[110:120]))
    print("y_train: {}".format(y_train[110:120]))
    print("x_test: {}".format(x_test[110:120]))
    print("y_test: {}".format(y_test[110:120]))


def macro_f1(y_true, y_pred):
    """Compute macro_f1 score give pred ys and true labels.

    We will change prob arys into one hot arys to ensure its correctness.

    Args:
        y_true: true labels, Tensor: (samples, max_class).
                Each sample has only 1., 0.s else.
                e.g. [[0., 1., 0., 0., 0., ..., 0.], [0.,, 0., 1., 0., ..., 0.], ... []]
        y_pred: model prediction outputs, Tensor: (samples, max_class).
                Each sample are probabilities across 20 classes, whose sum is near to 1.0.
                e.g. [[0.00232432, 0.100023, 0.678438, 0.0001, 0.0324132, ..., 0.0030], ... []]

    Returns:
        Macro-f1 value, Tensor.

    """
    # count appeared classes
    num_classes = np.sum(np.any(y_true, axis=0))

    def f1(precision_, recall_):
        return (2.0 * precision_ * recall_) / (precision_ + recall_ + np.finfo(float).eps)

    def precision(y_true, y_pred):
        true_positives = np.sum(y_true * y_pred, axis=0)  # axis=0: add by column
        pred_positives = np.sum(y_pred, axis=0)
        p = true_positives / (pred_positives + np.finfo(float).eps)
        return p

    def recall(y_true, y_pred):
        true_positives = np.sum(y_true * y_pred, axis=0)
        real_positives = np.sum(y_true, axis=0)
        r = true_positives / (real_positives + np.finfo(float).eps)
        return r

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f1_total = np.sum(f1(p, r))

    macrof1 = f1_total / num_classes
    return macrof1


class LossFunctions(object):
    def __init__(self, emoji_embedding_matrix, alpha=0.9):
        self.emoji_embedding_matrix = K.variable(emoji_embedding_matrix)
        self.embedding_size = int(self.emoji_embedding_matrix.shape[0])
        self.embedding_dim = int(self.emoji_embedding_matrix.shape[-1])
        self.batch_size = None
        self.x_n = K.l2_normalize(self.emoji_embedding_matrix, axis=-1)
        self.alpha = alpha

    def repeat_each(self, v, n):
        """repeat each element in tensor v n times on the second dim.
        [[1, 1, 1]          [[[1, 1, 1], [1, 1, 1], [1, 1, 1]]
         [2, 2, 2]   n=3     [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
         [3, 3, 3]   ===>    [[3, 3, 3], [3, 3, 3], [3, 3, 3]]
         [4, 4, 4]]          [[4, 4, 4], [4, 4, 4], [4, 4, 4]]]
        Args:
            v: tensor to repeat
            n: repeat n times on the second dim

        Returns:

        """
        vs = K.repeat(v, n)
        vs = K.reshape(vs, [self.batch_size, -1, self.embedding_dim])
        return vs

    def repeat_whole(self, v, n):
        """repeat tensor v n times on the second dim.
        [[1, 1, 1]          [[[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]
         [2, 2, 2]   n=3     [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]
         [3, 3, 3]   ===>    [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]]
         [4, 4, 4]]
        Args:
            v: tensor to repeat
            n: repeat n times on the first dim

        Returns:

        """
        vs = K.tile(v, [n, 1])
        vs = K.reshape(vs, [-1, self.embedding_size, self.embedding_dim])
        return vs

    def cosine_distance(self, y_pred, X_n):
        y_pred_n = K.l2_normalize(y_pred, axis=-1)
        y_preds_n = self.repeat_each(y_pred_n, self.embedding_size)
        xs_n = X_n
        return 1. - K.sum(xs_n * y_preds_n, axis=-1)

    def cosine_margin(self, y_true, y_pred):
        """Cosine margin loss function.

        Minimize the cosine distance between y and y_true, and maximize
        the minimum cosine distance between y and y_falses.
        formular: min {- s(y, y_true) + min[s(y, y_false)]}
        * s(y, y_true): cosine similarity function

        Args:
            y_true: true label (batch_size, emb_dim)
            y_pred: pred value (batch_size, emb_dim)

        Returns:
            mean cosine margin loss of this batch
        """
        self.batch_size = K.shape(y_pred)[0]
        X = self.repeat_whole(self.emoji_embedding_matrix, self.batch_size)
        X_n = self.repeat_whole(self.x_n, self.batch_size)
        y_trues = self.repeat_each(y_true, self.embedding_size)
        diff = K.sum(K.abs(y_trues - X), axis=-1)
        labels = K.less_equal(diff, K.epsilon())
        true_mask = labels
        false_mask = K.tf.logical_not(labels)
        cos_dises = self.cosine_distance(y_pred, X_n)
        true_dis = K.tf.boolean_mask(cos_dises, true_mask)  # 1d Tensor [?]
        false_dises = K.tf.boolean_mask(cos_dises, false_mask)
        false_dises = K.reshape(false_dises, [self.batch_size, -1])  # 2d Tensor [?, ?]
        min_false_dis = K.min(false_dises, axis=-1)  # 1d Tensor [?]
        ret1 = true_dis - min_false_dis
        ret = K.mean(ret1)
        # ret = K.sqrt(K.mean(ret1))
        return ret

    def cosine_margin_with_alpha(self, y_true, y_pred):
        """Cosine margin loss function.

        Minimize the cosine distance between y and y_true, and maximize
        the minimum cosine distance between y and y_falses.
        formular: min alpha * {- s(y, y_true) + (1 - alpha) * min[s(y, y_false)]}
        * s(y, y_true): cosine similarity function

        Args:
            y_true: true label (batch_size, emb_dim)
            y_pred: pred value (batch_size, emb_dim)

        Returns:
            mean cosine margin loss of this batch
        """
        self.batch_size = K.shape(y_pred)[0]
        X = self.repeat_whole(self.emoji_embedding_matrix, self.batch_size)
        X_n = self.repeat_whole(self.x_n, self.batch_size)
        y_trues = self.repeat_each(y_true, self.embedding_size)
        diff = K.sum(K.abs(y_trues - X), axis=-1)
        labels = K.less_equal(diff, K.epsilon())
        true_mask = labels
        false_mask = K.tf.logical_not(labels)
        cos_dises = self.cosine_distance(y_pred, X_n)
        true_dis = K.tf.boolean_mask(cos_dises, true_mask)  # 1d Tensor [?]
        false_dises = K.tf.boolean_mask(cos_dises, false_mask)
        false_dises = K.reshape(false_dises, [self.batch_size, -1])  # 2d Tensor [?, ?]
        min_false_dis = K.min(false_dises, axis=-1)  # 1d Tensor [?]
        ret1 = self.alpha * true_dis - (1.0 - self.alpha) * min_false_dis
        ret = K.mean(ret1)
        # ret = K.sqrt(K.mean(ret1))
        return ret


def cosine_similarity(y, X):
    """Compute cosine similarity between vector y and matrix X.

    Note: cosine_similarity = (x·y) /(‖x‖*‖y‖)   # ‖·‖ is L2 norm.

    Args:
        y: model predict vector. (dim, 1)
        X: emoji embedding matrix. (emoji_num, dim)

    Returns:
        cosine similarity between y and each row of X.
    """
    prod = np.dot(X, y)
    ones = np.ones_like(y)
    xlen = np.dot(np.square(X), ones)
    ylen = np.sum(np.square(y))
    ret = prod / np.sqrt(xlen * ylen)
    return ret


def cosine_distance(y_true, y_pred):
    """Compute cosine distance between y_true and y_pred.

    This is a loss function to minimize.
    Note: cosine distance = 1 - cosine_similarity

    Args:
        y_true: (batch_size, vector_dim) true vectors
        y_pred: (batch_size, vector_dim) pred vectors

    Returns:
        average cosine distance of y_true and y_pred
    """
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return K.mean(1 - K.sum(y_true * y_pred, axis=-1))


class CheckLoss(keras.callbacks.Callback):
    def __init__(self):
        pass

    def on_batch_begin(self, batch, logs=None):
        print("On batch begin!", self.model)

    def on_batch_end(self, batch, logs=None):
        print("On batch end!", self.model)


class MacroF1Classification(keras.callbacks.Callback):
    def __init__(self):
        self.macrof1s = 0.

    def on_epoch_end(self, epoch, logs={}):
        y_pred = np.asarray(self.model.predict(self.validation_data[0]))
        y_true_onehot = np.asarray(self.validation_data[1])
        y_pred_onehot = np.zeros_like(y_pred)
        y_pred_onehot[np.arange(y_pred.shape[0]), np.argmax(y_pred, axis=-1)] = 1.0
        self.macrof1s = macro_f1(y_true_onehot, y_pred_onehot)

        print(">> test dataset size: prediction: {}, true label: {}".format(y_pred_onehot.shape, y_true_onehot.shape))
        print(">> macro f1 after this epoch: {:.4f}\n".format(self.macrof1s * 100))
        return


class MacroF1Regression(keras.callbacks.Callback):
    def __init__(self, emoji_embedding_matrix, y_trues):
        """Compute macrof1 for regression model after each epoch.

        Args:
            emoji_embedding_matrix: 20 emojis' embedding matrix.
            y_trues: true labels for these data.
        """
        self.macrof1s = 0.
        self.emoji_embedding_matrix = emoji_embedding_matrix
        self.y_trues = y_trues

    def most_similar_emoji(self, ys):
        labels = []
        for y in ys:
            cos_sims = cosine_similarity(y, self.emoji_embedding_matrix)
            label = np.argmax(cos_sims, axis=-1)
            labels.append(label)
        return labels

    def on_epoch_end(self, epoch, logs={}):
        y_preds = np.asarray(self.model.predict(self.validation_data[0]))
        pred_emoji_idxes = self.most_similar_emoji(y_preds)
        y_trues_onehot = np.zeros(shape=(len(self.y_trues), 20))
        y_trues_onehot[np.arange(len(self.y_trues)), self.y_trues] = 1.0
        y_preds_onehot = np.zeros(shape=(len(pred_emoji_idxes), 20))
        y_preds_onehot[np.arange(len(pred_emoji_idxes)), pred_emoji_idxes] = 1.0

        self.macrof1s = macro_f1(y_trues_onehot, y_preds_onehot)

        print(">> test dataset size: prediction: {}, true label: {}".format(y_preds_onehot.shape, y_trues_onehot.shape))
        print(">> macro f1 after this epoch: {:.4f}\n".format(self.macrof1s * 100))

        # save y_preds to file
        out_file = "./output/eval/out.txt".replace("/", os.path.sep)
        with open(out_file, "w", encoding="utf-8") as fw:
            for y in pred_emoji_idxes:
                fw.write(str(y) + "\n")
        return

    def predict_classes(self, ys):
        pred_emoji_idxes = self.most_similar_emoji(ys)
        return pred_emoji_idxes


def argsort_martrix(m, inverse=True):
    if not isinstance(m, np.ndarray):
        m = np.array(m)
    print(m.shape)
    nrow, ncol = m.shape
    flat_m = m.flatten()
    sorted_idxes = np.argsort(flat_m)

    def oned_to_twod(idx, ncol):
        x = idx // ncol
        y = idx % ncol
        return x, y

    idx_pairs = [oned_to_twod(idx, ncol) for idx in sorted_idxes]
    if inverse is True:
        return idx_pairs[::-1]
    else:
        return idx_pairs


def similar_tweets(text_file):
    with open(text_file, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    vect = TfidfVectorizer(min_df=1)
    tfidf = vect.fit_transform(lines)
    cos_sim = tfidf * tfidf.T
    print("cos sim:")
    print(cos_sim.shape)
    sorted_idx_pairs = argsort_martrix(cos_sim.toarray(), inverse=True)
    return sorted_idx_pairs


def build_vocab(text_file):
    """Build vocabulary for text file.

    Train vocab covered 81.54% of trial vocab.

    Preprocess steps:
        1. lower case
        2. remove characters not int [a~z]

    Args:
        text_file: the file contains texts.

    Returns:
        vocab: dict: {w1: cnt1, w2: cnt2, ....}
    """

    class MySentences(object):
        def __init__(self, text_file_names):
            if isinstance(text_file_names, str):
                text_file_names = [text_file_names]
            self.file_names = text_file_names

        def __iter__(self):
            for file_name in self.file_names:
                with open(file_name, "r", encoding="utf-8") as fr:
                    for line in fr:
                        line = line.lower()
                        line = re.sub(r"[^a-z]", " ", line)
                        words = line.split()
                        for w in words:
                            yield w

    tweets = MySentences(text_file)
    vocab = Counter(tweets)
    return dict(vocab)


class LoadWord2VecData(object):
    def __init__(self, train_text_file, train_label_file, test_text_file, test_label_file, embedding_model_path,
                 train_size=None):
        self.train_text_file = train_text_file
        self.train_label_file = train_label_file
        self.test_text_file = test_text_file
        self.test_label_file = test_label_file
        self.train_size = train_size

        self.x_train, self.y_train, self.x_test, self.y_test = [], [], [], []
        self.vocab, self.word2idx = None, None
        self.emb_model_path = embedding_model_path
        self.emb_vocab_size, self.emb_size = 0, 0
        self.embedding_weights = None

        self.__load_data()
        self.__load_embedding()

    def __load_data(self):
        def preprocess(line):
            line = line.lower()
            line = re.sub(r"[^a-z]", " ", line)
            words = line.split()
            return words

        def split_texts(text_file):
            with open(text_file, "r") as fr:
                texts = []
                for line in fr:
                    words = preprocess(line)
                    texts.append(words)
            return texts

        def words2idxes(words_list):
            idxes = []
            for ws in words_list:
                word_idxes = [self.word2idx[w] for w in ws]
                idxes.append(word_idxes)
            return idxes

        # load x data
        train_words_list, test_words_list = split_texts(self.train_text_file), split_texts(self.test_text_file)
        train_words_list = train_words_list[:self.train_size]
        words = []
        for ws in train_words_list + test_words_list:
            for w in ws:
                words.append(w)
        vocab_tuple = Counter(words).most_common()  # sort by word freqency
        self.vocab = [w for w, cnt in vocab_tuple]
        self.word2idx = {w: idx + 1 for idx, w in enumerate(self.vocab)}  # idx will start from 1
        x_train, x_test = words2idxes(train_words_list), words2idxes(test_words_list)
        del train_words_list, test_words_list

        # load y data
        with open(self.train_label_file, "r") as fr:
            y_train = [int(label) for label in fr][:self.train_size]
        with open(self.test_label_file, "r") as fr:
            y_test = [int(label) for label in fr]

        self.x_train, self.x_test = np.array(x_train), np.array(x_test)
        self.y_train, self.y_test = np.array(y_train), np.array(y_test)

    def __load_embedding(self):
        def load_word2vec():
            # load embeddings first
            model = Word2Vec.load(self.emb_model_path)
            emb_vocab = model.wv.vocab
            emb_shape = (len(model.wv.vocab), len(model.wv[list(emb_vocab.keys())[0]]))
            return model.wv, emb_shape

        def build_embedding_matrix(word2idx, wv, embedding_size):
            emb_matrix = np.zeros(shape=(len(word2idx) + 1, embedding_size))  # 0: for unk
            added_idx = set()
            for w, idx in self.word2idx.items():
                if idx not in added_idx:
                    try:
                        emb_matrix[idx] = wv[w]
                        added_idx.add(idx)
                    except KeyError:  # word not found in embedding matrix will be all zeros
                        pass
            return emb_matrix

        # build needed embedding matrix
        wv, (self.emb_vocab_size, self.emb_size) = load_word2vec()
        self.embedding_weights = build_embedding_matrix(self.word2idx, wv, self.emb_size)


if __name__ == "__main__":
    print("Testing cosine margin:")
    y_pred = [0.2, -0.2, 0.2]
    y_true = [0., 1.2, 0.]
    emb_matrix = [[-1, 2.3, 3.], [5.2, 7., 0.], [-3.5, 2.1, -6.], [4., 0., 0.2]]
    emb_matrix = [[0, -1., 0.5], [0., -1., 0], [0, -1.5, 0.3], [0., 1.2, 0.]]
    lossfunc = LossFunctions(emoji_embedding_matrix=emb_matrix)
    print("cosin_margin: {}".format(lossfunc.cosine_margin(y_true, y_pred)))

