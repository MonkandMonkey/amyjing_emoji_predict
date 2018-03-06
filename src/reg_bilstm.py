"""Use pretrained Twitter embeddings and lstm neural network to predict emoji in tweets.

We posit that a emoji's embedding represents the contexts that it usually appears.
We first use the words' embeddings (in the tweet) to generate a vector which has the same dim as the embedding.
And then, we compute the cosine similarity between the vector and 20 emojis' embeddings, among which we choose
the emoji with highest cos sim as our predict label.

Hope this will work!

Pretrained embedding url: https://github.com/fvancesco/acmmm2016
"""
import logging

from keras.layers import Dense
from keras.layers import Embedding, LSTM, Bidirectional
from keras.models import Sequential

from src.nn_model import NeuralNetworkModel
from src.utils.utils import MacroF1Regression


class RegressionBilstmModel(NeuralNetworkModel):
    def __init__(self):
        logging.info("You are now using [regression BiLSTM] model.")
        # load data, text2seq2, pad sequences, load embeddings
        NeuralNetworkModel.__init__(self, pretrained_embedding=True, experiment_name="embbilstm_test")
        # overwrite parameters
        self.lstm_output_size = 300
        self.loss = "cosine_proximity"

        # load 20 emojis' embedding
        self.emoji_embedding_matrix = self.load_emoji_embeddings()

        # y to embedding
        self.y_to_embedding(self.emoji_embedding_matrix)

        # keep labels to compute macro f1
        self.y_trues = self.y_valid
        self.macro_f1 = MacroF1Regression(emoji_embedding_matrix=self.emoji_embedding_matrix, y_trues=self.y_trues)

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
            - Embedding: (word idxes) => (vocab_size, embedding_dims)
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

        # concanate the words' embeddings
        model.add(Bidirectional(LSTM(units=self.lstm_output_size), merge_mode="sum"))

        model.add(Dense(units=self.embedding_dim, activation="linear"))

        # compile model
        model.compile(loss=self.loss,  # mean_squared_error, cosine_distance, cosine_proximity
                      optimizer="adam",
                      metrics=["mse"])

        self.model = model
        logging.info("Model has been built.")
