"""Twitter emoji dataset for SemEval-2018 Task2——Emoji Predict.

Homepage: https://competitions.codalab.org/competitions/17344#participate

Lang: English dataset.
Each tweet contains only 1 emoji, and only the 20 most frequent emojis are considered.

Data format:
    X (string): tweet text extract out the emoji
    y (int from 0 to 19): emoji label
"""
import os
import re
import sys
import json
import logging

import numpy as np
import pandas as pd

from gensim.models import Word2Vec, KeyedVectors
from src.utils.utils import train_tokenizer


def load_json_data(filename):
    """Load json twitter data.

    [{"text": "hello world!", "label": 19}, {}, {}]

    Args:
        filename: json file name

    Returns:
        texts: []: twitter text list
        labels: []: twitter emoji label list (0~19)
    """
    with open(filename, "r", encoding="utf-8") as fr:
        temp_data = json.load(fr)
        texts, labels = [], []
        for d in temp_data:
            text = d["text"]
            label = d["label"]
            texts.append(text)
            labels.append(label)
    return texts, labels


class EmojiDataset(object):
    def __init__(self, data_dir="data/", num_words=None):
        """Emoji dataset class.

        Args:
            data_dir: your path to data dir.
        """
        # attributes
        self.__name__ = "twitter emoji dataset"
        self.__creator__ = "JingChen"
        self.__donor__ = "SemEval2018"
        self.__description__ = "balabala"
        self.__data_dir__ = data_dir
        self.original_file_path = {
            "train": self.__data_dir__ + "us_train.json",
            "valid": self.__data_dir__ + "us_valid.json",
            "test": self.__data_dir__ + "us_test.json",
        }
        self.file_description = {
            "train": "The train data set, not overlapping with valid and test data sets.\n"
                     "size: 466,233",
            "valid": "The validation data set, not overlapping with train data set.\n"
                     "size: 50,000",
            "test": "The test data set, not overlapping with train data set.\n"
                    "size: 50,000",
        }
        self.class_num = 20
        self.emoji_names = ["_red_heart_", "_smiling_face_with_hearteyes_", "_face_with_tears_of_joy_",
                            "_two_hearts_", "_fire_", "_smiling_face_with_smiling_eyes_",
                            "_smiling_face_with_sunglasses_", "_sparkles_", "_blue_heart_", "_face_blowing_a_kiss_",
                            "_camera_", "_United_States_", "_sun_", "_purple_heart_", "_winking_face_",
                            "_hundred_points_", "_beaming_face_with_smiling_eyes_", "_Christmas_tree_",
                            "_camera_with_flash_", "_winking_face_with_tongue_"]
        self.idx2emoji = {
            0: '❤', 1: '😍', 2: '😂', 3: '💕', 4: '🔥', 5: '😊', 6: '😎', 7: '✨', 8: '💙', 9: '😘', 10: '📷',
            11: '🇺🇸', 12: '☀', 13: '💜', 14: '😉', 15: '💯', 16: '😁', 17: '🎄', 18: '📸', 19: '😜'
        }

        assert len(self.idx2emoji) == self.class_num
        assert len(self.emoji_names) == self.class_num

        self.train_data, self.train_target = None, None
        self.valid_data, self.valid_target = None, None
        self.test_data, self.test_target = None, None

        self.tokenizer = None
        self.word_index = None
        self.word_counts = None
        self.num_words = num_words

    def load_data(self):
        """Load train and test dataset.

        Returns:
            (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
        """
        logging.info("Loading the [train, valid, test] dataset ...")

        self.train_data, self.train_target = load_json_data(self.original_file_path["train"])
        self.valid_data, self.valid_target = load_json_data(self.original_file_path["valid"])
        self.test_data, self.test_target = load_json_data(self.original_file_path["test"])

        return (self.train_data, self.train_target), \
               (self.valid_data, self.valid_target), \
               (self.test_data, self.test_target)

    def y_to_embedding(self, wv):
        """Change emoji label into its corresponding embedding. Used for training

        Args:
            wv: gensim word2vec model
        Returns:
            y_train, y_valid
        """
        y_train, y_valid = [], []
        if self.train_target is None or self.valid_target is None:
            raise ValueError("self.train_target, self.valid_target can't be None.")

        for emoji_idx in self.train_target:
            try:
                emb = wv[self.idx2emoji[emoji_idx]]
                y_train.append(emb)
            except KeyError:
                logging.info("Emoji: {} can't be found in wv.".format(emoji_idx))

        for emoji_idx in self.valid_target:
            try:
                emb = wv[self.idx2emoji[emoji_idx]]
                y_valid.append(emb)
            except KeyError:
                logging.info("Emoji: {} can't be found in wv.".format(emoji_idx))

        self.train_target, self.valid_target = y_train, y_valid
        return self.train_target, self.valid_target

    def build_tokenizer(self):
        """Build tokenizer fit on self.train_data.
        """
        if self.tokenizer is None:
            self.tokenizer = train_tokenizer(self.train_data, num_words=self.num_words)
        return self.tokenizer

    def texts2indexes(self):
        """Change word lists to index lists.
            num_words: max word idx to keep (by frequency)

        Returns:
            train index list
            test index list
            pred index list
        """
        if self.tokenizer is None:
            logging.error("Tokenizer must be built before texts2indexes!")
            sys.exit(0)
        train_idx_list = self.tokenizer.texts_to_sequences(self.train_data)
        valid_idx_list = self.tokenizer.texts_to_sequences(self.valid_data)
        test_idx_list = self.tokenizer.texts_to_sequences(self.test_data)
        return train_idx_list, valid_idx_list, test_idx_list

    def class_distribution(self):
        """Print the distribution on each class."""

        # categeorial distribution
        def class_distribution(labels, msg="this set"):
            """labels: [12, 9, 0, 7, 2, ...]
            """
            classes = {}
            for label in labels:
                if label in classes:
                    classes[label] += 1
                else:
                    classes[label] = 1

            num = len(labels)
            sorted_classes = sorted(classes, key=classes.get, reverse=True)
            logging.info("\nClass distribution on {} :".format(msg))
            for t in sorted_classes:
                p = classes[t] / num
                logging.info("[{}]: {} {:.4%}".format(t, classes[t], p))

        class_distribution(self.train_target, msg="train set")
        class_distribution(self.valid_target, msg="valid set")
        class_distribution(self.test_target, msg="test set")

    def lengths_distribution(self):
        """Print the twitter lengths(by words) distribution."""

        def get_word_num(tweet):
            tweet = re.sub(r"([^a-z0-9])", r" \1", tweet)
            words = tweet.split()
            return len(words)

        def len_distrib(tweets, msg="this set"):
            lens = []
            for tweet in tweets:
                len_ = get_word_num(tweet)
                lens.append(len_)

            print("\nLengths distribution on {} :".format(msg))
            print(pd.Series(lens).describe())

        len_distrib(self.train_data, msg="train set")
        len_distrib(self.valid_data, msg="valid set")
        len_distrib(self.test_data, msg="test set")

    def vocab_information(self, p=0.8, max_word_idx=None, cnt=2):
        if self.tokenizer is None:
            self.tokenizer = train_tokenizer(self.train_data)

        self.word_index = self.tokenizer.word_index
        self.word_counts = self.tokenizer.word_counts
        idx2word = {idx: w for w, idx in self.word_index.items()}
        print("vocab size: {:,}".format(len(self.word_index)))
        words_num = 0
        for w, idx in self.word_index.items():
            words_num += self.word_counts[w]

        vocab_size = len(self.word_index)
        if max_word_idx is not None:
            num = 0
            for idx in range(1, vocab_size + 1):
                num += self.word_counts[idx2word[idx]]
                if idx >= max_word_idx:
                    p = num / words_num
                    print("word idx [{:,}] account for: {:.2%} in vocab.".format(max_word_idx, p))
                    break
        elif p is not None:
            num = words_num * p
            for idx in range(1, vocab_size + 1):
                num -= self.word_counts[idx2word[idx]]
                if num <= 0:
                    print("{:.2%} boards {:,} in vocab.".format(p, idx))
                    break
        elif cnt is not None:
            for idx in range(1, vocab_size + 1):
                num = 0
                t = self.word_counts[idx2word[idx]]
                if t <= cnt:
                    p = num / words_num
                    print("count lager than {} accounts for {:.2%} in vocab.".format(cnt, p))
                    break
        else:
            raise ValueError("p and max_word_idx can't both be None.")


class Embeddings(object):
    """Build embedding matrix for texts.

    With the word2idx, and the pretrained embedding matrix, we lookup our needed embeddings from the matrix
    to form a smaller embedding matrix for futher training.
    """

    def __init__(self,
                 word2idx,
                 num_words=None,
                 embedding_model_path="data/word2vec/model_swm_300-6-10-low.w2v"):

        if embedding_model_path is None:
            raise ValueError("The embedding model's path is needed.")

        if word2idx is None:
            raise ValueError("The word2idx param can't be None, it is used for decide which words "
                             "will be kept in the final embedding matrix.")
        self.word2idx = None
        self.num_words = num_words
        if self.num_words is not None:
            self.word2idx = {w: i for w, i in word2idx.items() if i < self.num_words}
        else:
            self.word2idx = {w: i for w, i in word2idx.items()}

        self.embedding_model_path = embedding_model_path
        self.emoji2vec_model_path = "data/word2vec/emoji2vec.bin"
        self.wv, self.embedding_matrix = None, None
        self.embedding_shape = None

        # load the whole pretrained word2vec model
        self.load_word2vec()

    def load_word2vec(self):
        # load embeddings first
        if self.embedding_model_path[-3:] == "bin":
            model = KeyedVectors.load_word2vec_format(self.embedding_model_path, binary=True)
        elif self.embedding_model_path[-7:-4] == "low" or self.embedding_model_path[-3:] == "txt":
            model = KeyedVectors.load_word2vec_format(self.embedding_model_path, binary=False)
        else:
            model = Word2Vec.load(self.embedding_model_path)
        emb_vocab = model.wv.vocab
        emb_shape = (len(model.wv.vocab), len(model.wv[list(emb_vocab.keys())[0]]))
        self.wv, self.embedding_shape = model.wv, emb_shape

    def load_embedding(self):
        def build_embedding_matrix(word2idx, wv, embedding_size):
            emb_matrix = np.zeros(shape=(len(word2idx) + 1, embedding_size))  # 0: for unk
            added_idx = set()
            for w, idx in word2idx.items():
                if idx not in added_idx:
                    try:
                        emb_matrix[idx] = wv[w]
                        added_idx.add(idx)
                    except KeyError:  # word not found in embedding matrix will be all zeros
                        pass
            return emb_matrix

        # build needed embedding matrix
        self.embedding_matrix = build_embedding_matrix(self.word2idx, self.wv, self.embedding_shape[-1])
        return self.embedding_matrix

    def load_emoji_embedding(self):
        idx2emoji = {
            0: '❤', 1: '😍', 2: '😂', 3: '💕', 4: '🔥', 5: '😊', 6: '😎', 7: '✨', 8: '💙', 9: '😘', 10: '📷',
            11: '🇺🇸', 12: '☀', 13: '💜', 14: '😉', 15: '💯', 16: '😁', 17: '🎄', 18: '📸', 19: '😜'
        }
        emoji_num = len(idx2emoji)
        emoji_embedding_matrix = np.zeros(shape=(emoji_num, self.embedding_shape[-1]))
        for idx in range(emoji_num):
            emoji = idx2emoji[idx]
            try:
                emb = self.wv[emoji]
                emoji_embedding_matrix[idx] = emb
            except KeyError:
                raise KeyError("Emoji: [{}] embedding can't be found!".format(emoji))
        return emoji_embedding_matrix

    def load_emoji2vec_embedding(self):
        idx2emoji = {
            0: '❤️', 1: '😍', 2: '😂', 3: '💕', 4: '🔥', 5: '😊', 6: '😎', 7: '✨', 8: '💙', 9: '😘', 10: '📷',
            11: '🇺🇸', 12: '☀️', 13: '💜', 14: '😉', 15: '💯', 16: '😁', 17: '🎄', 18: '📸', 19: '😜'
        }
        model = KeyedVectors.load_word2vec_format(self.emoji2vec_model_path, binary=True)
        emoji2vec = model.wv
        emoji_num = len(idx2emoji)
        print("self.embedding_shape: {}".format(self.embedding_shape))
        emoji_embedding_matrix = np.zeros(shape=(emoji_num, 300))
        for idx in range(emoji_num):
            emoji = idx2emoji[idx]
            try:
                emb = emoji2vec[emoji]
                emoji_embedding_matrix[idx] = emb
            except KeyError:
                raise KeyError("Emoji: [{}] embedding can't be found!".format(emoji))
        return emoji_embedding_matrix
