import logging
import os
from config.paths import EXPERIMENTS_LOG_PATH


def setup_logger(script_type):
    log_directory = os.path.dirname(EXPERIMENTS_LOG_PATH)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    logger = logging.getLogger(script_type)
    logger.setLevel(logging.INFO)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(EXPERIMENTS_LOG_PATH)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
