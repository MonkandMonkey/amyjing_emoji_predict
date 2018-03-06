"""Use pretrained Twitter embeddings and lstm and dense neural network to predict emoji in tweets.

We posit that a emoji's embedding represents the contexts that it usually appears.
We first use the words' embeddings (in the tweet) to generate a vector which has the same dim as the embedding.
And then, we compute the cosine similarity between the vector and 20 emojis' embeddings, among which we choose
the emoji with highest cos sim as our predict label.

Hope this will work!

Pretrained embedding url: https://github.com/fvancesco/acmmm2016
"""
import os
import time
import logging

import numpy as np
from keras.layers import Dense
from keras.layers import Embedding, LSTM
from keras.models import Sequential, load_model
from keras.preprocessing import sequence
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from src.utils.utils import MacroF1Regression, LossFunctions
from src.utils.emoji_dataset import EmojiDataset, Embeddings


class RegressionLstmModel(object):
    """
    Regression approach 2-layer LSTM model.
    """

    def __init__(self, opt="origin"):
        logging.info("You are now using [regression 2-layer LSTM] model.")

        # remember to check file paths
        self.experiment_name = "emblstm-l2-alpha0.99-test"
        self.output_file = "output/eval/{}.out.txt".format(self.experiment_name)
        self.model_file = "output/models/{}.h5".format(self.experiment_name)
        self.model_pic_file = "output/models/{}.png".format(self.experiment_name)

        # set parameters
        self.num_classes = 20
        self.num_words = 58205
        self.max_len = 20
        self.batch_size = 128
        self.epochs = 20

        self.lstm_size = 300
        self.hidden_size = 300

        self.alpha = 0.99

        self.model = None

        # load data
        emoji_dataset = EmojiDataset(num_words=self.num_words)
        (self.x_train, self.y_train), (self.x_valid, self.y_valid), (self.x_test, self.y_test) \
            = emoji_dataset.load_data()
        # keep labels to compute macro f1
        self.y_trues = self.y_valid

        # texts to sequences
        self.tokenizer = emoji_dataset.build_tokenizer()
        self.x_train, self.x_valid, self.x_test = emoji_dataset.texts2indexes()

        # pad sequences
        logging.info("Padding sequences ...")
        self.x_train = sequence.pad_sequences(self.x_train, maxlen=self.max_len)
        self.x_valid = sequence.pad_sequences(self.x_valid, maxlen=self.max_len)
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=self.max_len)

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

        # load emoji embedding matrix
        self.emoji_embedding_matrix = embeddings.load_emoji_embedding()
        logging.debug("emoji_embedding_matrix: {}".format(self.emoji_embedding_matrix.shape))

        # y to embedding
        logging.info('Change emoji label to its embedding ...')
        self.y_train, self.y_valid = emoji_dataset.y_to_embedding(embeddings.wv)
        self.y_train, self.y_valid, self.y_test = np.array(self.y_train), np.array(self.y_valid), np.array(self.y_test)

        logging.info('x_train shape: {}'.format(self.x_train.shape))
        logging.info('y_train shape: {}'.format(self.y_train.shape))
        logging.info('x_valid shape: {}'.format(self.x_valid.shape))
        logging.info('y_valid shape: {}'.format(self.y_valid.shape))
        logging.info('x_test shape: {}'.format(self.x_test.shape))
        logging.info('y_test shape: {}'.format(self.y_test.shape))

    def build_model(self):
        """Build Embedding Mlp model.

        Model structure:
            - Input:
            - Embedding with dropout: (word idxes) => (vocab_size, embedding_dims)
            - Dense with dropout: activation: relu (hidden layer)
        """
        logging.info('Build model ...')
        model = Sequential()

        # embedding layer initilized with pretrained embedding
        model.add(Embedding(self.vocab_size + 1,  # due to mask_zero
                            self.embedding_dim,
                            input_length=self.max_len,
                            weights=[self.embedding_matrix],
                            trainable=True))
        model.add(LSTM(units=self.lstm_size, return_sequences=True))
        model.add(LSTM(units=self.lstm_size))
        model.add(Dense(units=self.hidden_size, activation="linear"))

        # compile model
        lossfunc = LossFunctions(emoji_embedding_matrix=self.emoji_embedding_matrix, alpha=self.alpha)
        model.compile(loss=lossfunc.cosine_margin_with_alpha,  # "cosine_proximity"
                      optimizer="adam",
                      metrics=["mse"])

        self.model = model
        logging.info("Model has been built.")

    def train_model(self):
        logging.info("Start training ...")
        early_stopping = EarlyStopping(monitor="val_loss", patience=3)
        # checkpoint
        filepath = "output/models/" + self.experiment_name + ".weights-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath)
        mac_f1 = MacroF1Regression(emoji_embedding_matrix=self.emoji_embedding_matrix, y_trues=self.y_trues)
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

    def load_model(self, model_path, custom_objects=None):
        """
        Load trained 2-layer lstm model. 
        If the model uses l2 loss function, you need to pass custom_object.
        """
        self.model = load_model(model_path, custom_objects=custom_objects)
