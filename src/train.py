"""
Train classification approach models: 2-layer lstm and bilstm
Train regression approach models: 2-layer lstm and bilstm

"""

from src.cls_lstm import ClassificationLstmModel
from src.cls_bilstm import ClassificationBilstmModel
from src.vs_bilstm import VectorSimilarityBilstmModel
from src.vs_lstm import VectorSimilarityLstmModel


def vs_lstm():
    """ regression 2-layer lstm model """
    model = VectorSimilarityLstmModel()
    model.build_model()
    model.plot_model_architecture()
    model.model.summary()
    model.train_model()
    model.save_model()


def vs_bilstm():
    """ regression bilstm model """
    model = VectorSimilarityBilstmModel()
    model.build_model()
    model.plot_model_architecture()
    model.model.summary()
    model.train_model()
    model.save_model()


def cls_lstm():
    """ classification 2-layer lstm model """
    model = ClassificationLstmModel()
    model.build_model()
    model.plot_model_architecture()
    model.model.summary()
    model.train_model()
    model.save_model()


def cls_bilstm():
    """ regression bilstm model """
    model = ClassificationBilstmModel()
    model.build_model()
    model.plot_model_architecture()
    model.model.summary()
    model.train_model()
    model.save_model()
