import logging
import sys

def get_my_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(levelname)7s]%(filename)12s: %(message)s")

    streamHandler = logging.StreamHandler(sys.stdout)
    streamHandler.setFormatter(formatter)

    logger.addHandler(streamHandler)
    return logger

