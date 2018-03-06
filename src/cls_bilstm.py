#! /usr/bin/env python
"""Use BiLSTM to predict emoji in tweets.

Different num of LSTM layers may be used.
"""
import logging

from keras.layers import Activation, Dense
from keras.layers import Embedding, LSTM, Bidirectional

from keras.models import Sequential

from src.nn_model import NeuralNetworkModel
from src.utils.utils import MacroF1Classification


class ClassificationBilstmModel(NeuralNetworkModel):
    def __init__(self):
        """
        LSTM model.
        Args:
            opt: dataset options: origin, small, balanced_large, balanced_small
        """
        logging.info("You are now using [classification BiLSTM] model.")
        # load data, text2seq2, pad sequences, load embeddings
        NeuralNetworkModel.__init__(self, pretrained_embedding=True, experiment_name="cls_bilstm")
        # overwrite parameters
        self.lstm_output_size = 300
        self.loss = "categorical_crossentropy"

        # y to embedding
        self.y_to_categorial()

        # compute macro f1 after each epoch
        self.macro_f1 = MacroF1Classification()

        logging.info('x_train shape: {}'.format(self.x_train.shape))
        logging.info('y_train shape: {}'.format(self.y_train.shape))
        logging.info('x_valid shape: {}'.format(self.x_valid.shape))
        logging.info('y_valid shape: {}'.format(self.y_valid.shape))
        logging.info('x_test shape: {}'.format(self.x_test.shape))
        logging.info('y_test shape: {}'.format(self.y_test.shape))

    def build_model(self):
        """Build BiLstm  model.

        Model structure:
            - Input:
            - Embedding with dropout: (word idxes) => (vocab_size, embedding_dims)
            - Bidirectional(LSTM)
            - Dense with dropout: activation: relu (hidden layer)
            - Softmax: convert vectors into 20 - d classes
        """
        logging.info("Building model ...")
        model = Sequential()

        if self.use_pretrained_embedding:
            embedding_layer = Embedding(self.vocab_size + 1,  # due to mask_zero
                                        self.embedding_dim,
                                        input_length=self.max_len,
                                        weights=[self.embedding_matrix],
                                        trainable=True)
        else:
            embedding_layer = Embedding(input_dim=self.num_words,
                                        output_dim=self.embedding_dim,
                                        input_length=self.max_len)

        model.add(embedding_layer)
        model.add(Bidirectional(LSTM(units=self.lstm_output_size), merge_mode="sum"))

        # output layer:
        model.add(Dense(self.num_classes))
        model.add(Activation("softmax"))

        # compile model
        model.compile(loss=self.loss,
                      optimizer="adam",
                      metrics=['accuracy'])

        self.model = model
        logging.info("Model has been built.")
