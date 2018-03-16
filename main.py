"""main function.

Start from here.
"""
from src.utils.log_utils import config_log
import src.train as train
import src.predict as predict


def main():
    config_log()
    train.vs_lstm()
    # .reg_lstm()


if __name__ == "__main__":
    main()
