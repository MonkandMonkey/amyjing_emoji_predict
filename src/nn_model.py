#! /usr/bin/env python
"""Abstract class for nn models.

class MyModel(NNModel):
"""
import sys
import time
import numpy as np
import logging

from keras.models import load_model
from keras.preprocessing import sequence
from keras.utils import plot_model, to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

from src.utils.emoji_dataset import EmojiDataset, Embeddings, build_tokenizer, texts2indexes, y_to_embedding
from src.utils.utils import ClassificationMacroF1, VecorSimilarityMacroF1


class NeuralNetworkModel(object):
    def __init__(self, pretrained_embedding=True, experiment_name="no_name"):
        """
        Abstract neural network model.
        Args:
           pretrained_embedding: load pretrained embeddings or not
           experiment_name: your experiment name
        """
        self.use_pretrained_embedding = pretrained_embedding
        self.experiment_name = experiment_name

        logging.info("Your experiment name is: [{}]".format(self.experiment_name))
        # remember to check file paths
        self.output_file = "output/eval/{}.out.txt".format(self.experiment_name)
        self.model_file = "output/models/{}.h5".format(self.experiment_name)
        self.model_pic_file = "output/models/{}.png".format(self.experiment_name)
        self.checkpoint_filepath = "output/models/" + self.experiment_name + ".weights-{epoch:02d}-{val_loss:.2f}.hdf5"
        # set common parameters
        self.num_classes = 20
        self.num_words = 58205
        self.max_len = 20
        self.embedding_dim = 300
        self.epochs = 20
        self.batch_size = 128

        self.loss = None  # str, or a fucntion
        self.macro_f1 = None  # compute macro f1 after each epoch
        self.model = None
        self.x_train, self.x_valid, self.x_test = None, None, None
        self.y_train, self.y_valid, self.y_test = None, None, None
        self.embedding_obj, self.embedding_matrix, self.embedding_dim, self.vocab_size = None, None, None, None

        # load data, text 2 sequences, padding sequences, load pretrained embeddings
        (self.x_train, self.y_train), (self.x_valid, self.y_valid), (self.x_test, self.y_test) = self.load_data()
        self.tokenizer = self.texts_2_sequences()
        self.pad_sequences()
        if self.use_pretrained_embedding:
            self.load_embeddings()

    def load_data(self):
        # load data
        logging.info("Loading data ...")
        emoji_dataset = EmojiDataset(num_words=self.num_words)
        return emoji_dataset.load_data()

    def texts_2_sequences(self):
        # texts to sequences
        tokenizer = build_tokenizer(self.x_train, self.num_words)
        self.x_train = texts2indexes(tokenizer, self.x_train)
        self.x_valid = texts2indexes(tokenizer, self.x_valid)
        self.x_test = texts2indexes(tokenizer, self.x_test)
        return tokenizer

    def pad_sequences(self):
        # pad sequences
        logging.info("Padding sequences ...")
        self.x_train = sequence.pad_sequences(self.x_train, maxlen=self.max_len)
        self.x_valid = sequence.pad_sequences(self.x_valid, maxlen=self.max_len)
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=self.max_len)

    def y_to_categorial(self):
        # Convert class vectors to binary class matrices.
        # return type: np.array()
        self.y_train = to_categorical(self.y_train, self.num_classes)
        self.y_valid = to_categorical(self.y_valid, self.num_classes)
        self.y_test = np.array(self.y_test)

    def y_to_embedding(self, emoji_embeddings):
        # y to embedding
        logging.info('Change emoji label to its embedding ...')
        self.y_train = y_to_embedding(emoji_embeddings, self.y_train)
        self.y_valid = y_to_embedding(emoji_embeddings, self.y_valid)
        self.y_train, self.y_valid, self.y_test = np.array(self.y_train), np.array(self.y_valid), np.array(self.y_test)

    def load_embeddings(self):
        # load embedding matrix
        # load word2vec model
        logging.info("Load word2vec model ...")
        embedding_model_path = "data/word2vec/model_swm_300-6-10-low.w2v"
        self.embedding_obj = Embeddings(word2idx=self.tokenizer.word_index,
                                        num_words=self.num_words,
                                        embedding_model_path=embedding_model_path)

        # load needed embedding matrix
        logging.info("Build needed embedding matrix ...")
        self.embedding_matrix = self.embedding_obj.load_embedding()
        self.vocab_size = self.embedding_matrix.shape[0] - 1
        self.embedding_dim = self.embedding_matrix.shape[1]
        logging.info("embedding shape: {}".format(self.embedding_matrix.shape))

    def load_emoji_embeddings(self):
        # load emoji embedding matrix
        emoji_embedding_matrix = self.embedding_obj.load_emoji_embedding()
        logging.debug("emoji_embedding_matrix shape: {}".format(emoji_embedding_matrix.shape))
        return emoji_embedding_matrix

    # def build_model(self):
    #     """Implement it in your class.
    #     """
    #     return

    def train_model(self):
        logging.info("Start training ...")
        early_stopping = EarlyStopping(monitor="val_loss", patience=3)

        checkpoint = ModelCheckpoint(self.checkpoint_filepath)
        start = time.time()
        if self.macro_f1:
            cbks = [self.macro_f1, early_stopping, checkpoint]
        else:
            cbks = [early_stopping, checkpoint]

        self.model.fit(self.x_train, self.y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_data=(self.x_valid, self.y_valid),
                       verbose=2,
                       callbacks=cbks)
        stop = time.time()
        logging.info("Finish training.\n")
        logging.info("Training used {} s.".format(stop - start))

    def predict(self, valid_output_file=None, test_output_file=None):
        if valid_output_file is None:
            valid_output_file = "output/eval/" + self.experiment_name + ".valid.txt"
        if test_output_file is None:
            test_output_file = "output/eval/" + self.experiment_name + ".test.txt"

        if isinstance(self.macro_f1, MacroF1Classification):
            valid_outputs = self.model.predict_classes(self.x_valid)
            test_outputs = self.model.predict_classes(self.x_test)
        elif isinstance(self.macro_f1, MacroF1Regression):
            y_pred_valid = self.model.predict(self.x_valid)
            y_pred_test = self.model.predict(self.x_test)
            valid_outputs = self.macro_f1.predict_classes(y_pred_valid)
            test_outputs = self.macro_f1.predict_classes(y_pred_test)
        else:
            logging.error("Can't judge classification and vector similarity!")
            sys.exit(0)

        logging.info("valid prediction: {}".format(len(valid_outputs)))
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
        # if you use custom loss function, you need pass custom_objects to load_model()
        # e.g. load_model(model_path, custom_objects={"cosine_margin_with_alpha":self.loss})
        self.model = load_model(model_path, custom_objects=custom_objects)
