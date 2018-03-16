# amyjing_emoji_predict
AmyJing at SemEval 2018 Task 2: Multilingual Emoji Prediction

# requirements
python 3
keras >= 2.0.6
tensorflow >= 1.2.1

# usage
python main.py

model files:
- nn_model.py: super class for each model, common experiment settings are here, you can overwrite param settings within each model's class in init.
- cls_bislstm.py: classification bilstm model
- cls_lstm.py: classification 2-layered lstm model
- vs_bilstm.py: vector similarity bilstm model
- vs_lstm.py: vector similarity 2-layered lstm model

english preptrained embedding:
- https://github.com/fvancesco/acmmm2016

datasets:
- train set: 466,233
- valid set: 50,000
- test  set: 50,000


