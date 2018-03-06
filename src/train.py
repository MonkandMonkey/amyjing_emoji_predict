"""
Train classification approach models: 2-layer lstm and bilstm
Train regression approach models: 2-layer lstm and bilstm

"""

from src.cls_lstm import ClassificationLstmModel
from src.cls_bilstm import ClassificationBilstmModel
from src.reg_bilstm import RegressionBilstmModel
from src.reg_lstm import RegressionLstmModel


def reg_lstm():
    """ regression 2-layer lstm model """
    model = RegressionLstmModel()
    model.build_model()
    model.plot_model_architecture()
    model.model.summary()
    model.train_model()
    model.save_model()


def reg_bilstm():
    """ regression bilstm model """
    model = RegressionBilstmModel()
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
