"""Use ogging tool."""
import os
import yaml
import logging
from logging.config import dictConfig


def load_logger():
    """Use logging tool.

    Config file is 'data/logging.conf.yaml'

    Returns:
        Root logger.
    """
    config_file = "config/logging.conf.yaml"
    with open(config_file, "r") as fr:
        config = yaml.load(fr.read())
    dictConfig(config)
    logger = logging.getLogger("root")

    logger.info("Your current working path is: {}".format(os.getcwd()))
    logger.info("You can now use your wonderful logging tool!")
    return logger


def config_log():
    """Use logging tool.

    Config file is 'data/logging.conf.yaml'

    Returns:
        Root logger.
    """
    config_file = "config/logging.conf.yaml"
    with open(config_file, "r") as fr:
        config = yaml.load(fr.read())
    dictConfig(config)
    print("Log config loaded.")
