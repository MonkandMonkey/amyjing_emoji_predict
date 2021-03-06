"""
Predict using trained model.
"""
from src.cls_bilstm import ClassificationBilstmModel
from src.cls_lstm import ClassificationLstmModel
from src.vs_bilstm import VectorSimilarityBilstmModel
from src.vs_lstm import VectorSimilarityLstmModel


def reg_lstm():
    """ regression 2-layer lstm  """
    model = VectorSimilarityLstmModel()
    model.load_model(model_path="")
    model.model.summary()
    model.predict()


def reg_bilstm():
    """ regression bilstm  """
    model = VectorSimilarityBilstmModel()
    model.load_model(model_path="")
    model.model.summary()
    model.predict()


def cls_lstm():
    """ classification 2-layer lstm  """
    model = ClassificationLstmModel()
    model.load_model(model_path="output/models/cls_lstm.weights-00-2.13.hdf5")
    model.model.summary()
    model.predict(valid_output_file="output/eval/cls_lstm.00.valid.out",
                  test_output_file="output/eval/cls_lstm.00.test.out")


def cls_bilstm():
    """ classification bilstm  """
    model = ClassificationBilstmModel()
    model.load_model(model_path="")
    model.model.summary()
    model.predict()
