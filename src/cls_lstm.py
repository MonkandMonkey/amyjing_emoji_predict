#! /usr/bin/env python
"""Use LSTM to predict emoji in tweets.

Different num of LSTM layers may be used.
"""
import os
import time
import numpy as np
import logging

from keras.layers import Activation, Dense
from keras.layers import Embedding, LSTM

from keras.models import Sequential, load_model
from keras.preprocessing import sequence
from keras.utils import plot_model, to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

from src.utils.utils import MacroF1Classification
from src.utils.emoji_dataset import EmojiDataset, Embeddings


class ClassificationLstmModel(object):
    def __init__(self, opt="origin", pretrained_embedding=True):
        """
        LSTM model.
        Args:
            opt: dataset options: origin, small, balanced_large, balanced_small
        """
        logging.info("You are now using [classification 2-layer LSTM] model.")

        self.use_pretrained_embedding = pretrained_embedding

        # remember to check file paths
        self.experiment_name = "cls_lstm"

        self.output_file = "output/eval/{}.out.txt".format(self.experiment_name)
        self.model_file = "output/models/{}.h5".format(self.experiment_name)
        self.model_pic_file = "output/models/{}.png".format(self.experiment_name)

        # set parameters
        self.num_classes = 20

        self.num_words = 58205
        self.max_len = 20
        self.batch_size = 128
        self.embedding_dim = 300

        self.lstm_output_size = 300

        self.epochs = 20

        self.model = None

        # embedding layer
        self.embedding_layer = None

        # load data
        logging.info("Loading data ...")
        emoji_dataset = EmojiDataset(num_words=self.num_words)
        (self.x_train, self.y_train), (self.x_valid, self.y_valid), (self.x_test, self.y_test) \
            = emoji_dataset.load_data()

        # texts to sequences
        self.tokenizer = emoji_dataset.build_tokenizer()
        self.x_train, self.x_valid, self.x_test = emoji_dataset.texts2indexes()

        # pad sequences
        logging.info("Padding sequences ...")
        self.x_train = sequence.pad_sequences(self.x_train, maxlen=self.max_len)
        self.x_valid = sequence.pad_sequences(self.x_valid, maxlen=self.max_len)
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=self.max_len)

        # Convert class vectors to binary class matrices.
        self.y_train = to_categorical(self.y_train, self.num_classes)
        self.y_valid = to_categorical(self.y_valid, self.num_classes)
        self.y_test = np.array(self.y_test)

        # load embedding matrix
        if self.use_pretrained_embedding is True:
            # load word2vec model
            logging.info("Load word2vec model ...")
            embedding_model_path = "data/word2vec/model_swm_300-6-10-low.w2v"
            embeddings = Embeddings(word2idx=self.tokenizer.word_index,
                                    num_words=self.num_words,
                                    embedding_model_path=embedding_model_path)

            # load needed embedding matrix
            logging.info("Build needed embedding matrix ...")
            self.embedding_matrix = embeddings.load_embedding()
            self.vocab_size = self.embedding_matrix.shape[0] - 1
            self.embedding_dim = self.embedding_matrix.shape[1]
            logging.info("embedding shape: {}".format(self.embedding_matrix.shape))

        logging.info('x_train shape: {}'.format(self.x_train.shape))
        logging.info('y_train shape: {}'.format(self.y_train.shape))
        logging.info('x_valid shape: {}'.format(self.x_valid.shape))
        logging.info('y_valid shape: {}'.format(self.y_valid.shape))
        logging.info('x_test shape: {}'.format(self.x_test.shape))
        logging.info('y_test shape: {}'.format(self.y_test.shape))

    def build_model(self):
        """Build sequential Conv1D model.

        Model structure:
            - Input:
            - Embedding with dropout: (word idxes) => (vocab_size, embedding_dims)
            - GRU
            - GRU
            - Dense with dropout: activation: relu (hidden layer)
            - Softmax: convert vectors into 20 - d classes
        """
        logging.info("Building model ...")
        model = Sequential()

        if self.use_pretrained_embedding:
            self.embedding_layer = Embedding(self.vocab_size + 1,  # due to mask_zero
                                             self.embedding_dim,
                                             input_length=self.max_len,
                                             weights=[self.embedding_matrix],
                                             trainable=False)
        else:
            self.embedding_layer = Embedding(input_dim=self.num_words,
                                             output_dim=self.embedding_dim,
                                             input_length=self.max_len)

        model.add(self.embedding_layer)
        model.add(LSTM(units=self.lstm_output_size, return_sequences=True))
        model.add(LSTM(units=self.lstm_output_size))

        # output layer:
        model.add(Dense(self.num_classes))
        model.add(Activation("softmax"))

        # compile model
        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=['accuracy'])

        self.model = model
        logging.info("Model has been built.")

    def train_model(self):
        logging.info("Start training ...")
        early_stopping = EarlyStopping(monitor="val_loss", patience=3)
        mac_f1 = MacroF1Classification()
        filepath = "output/models/" + self.experiment_name + ".weights-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath)
        start = time.time()
        self.model.fit(self.x_train, self.y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_data=(self.x_valid, self.y_valid),
                       verbose=2,
                       callbacks=[mac_f1, early_stopping, checkpoint])
        stop = time.time()
        logging.info("Finish training.\n")
        logging.info("Training used {} s.".format(stop - start))

    def predict(self, valid_output_file=None, test_output_file=None):
        if valid_output_file is None:
            valid_output_file = "output/eval/" + self.experiment_name + ".valid.txt"
        if test_output_file is None:
            test_output_file = "output/eval/" + self.experiment_name + ".test.txt"

        valid_outputs = self.model.predict_classes(self.x_valid)
        logging.info("valid prediction: {}".format(len(valid_outputs)))
        test_outputs = self.model.predict_classes(self.x_test)
        logging.info("test prediction: {}".format(len(test_outputs)))

        with open(valid_output_file, "w") as fw:
            for o in valid_outputs:
                fw.write(str(o) + "\n")
        with open(test_output_file, "w") as fw:
            for o in test_outputs:
                fw.write(str(o) + "\n")

        logging.info("valid predictions have been saved to: {}".format(valid_output_file))
        logging.info("test predictions have been saved to: {}".format(test_output_file))

    def save_model(self, filename=None):
        if filename is None:
            filename = self.model_file
        self.model.save(filename)
        logging.info("trained model has been saved into {}".format(filename))

    def plot_model_architecture(self, filename=None):
        if filename is None:
            filename = self.model_pic_file
        plot_model(self.model, to_file=filename)

    def load_model(self, model_path):
        self.model = load_model(model_path)
