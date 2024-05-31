import logging
from utils import api_logger

class AdminLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)

    def handle(self, record):
        api_logger.log_to_admin_logger(record)

        # Call the parent handler to log to the file
        super().handle(record)

loggers = {}

def create_logger(name, filename, level, format_str):
    logger = AdminLogger(name, level)  # Use AdminLogger instead of logging.Logger
    logger.setLevel(level)
    formatter = logging.Formatter(format_str)

    file_handler = logging.FileHandler(filename, mode='a')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    loggers[name] = logger

# Create loggers
create_logger('info_logger', 'logs/info.log', logging.INFO, '%(asctime)s %(message)s')
create_logger('warn_logger', 'logs/warn.log', logging.WARNING, '%(asctime)s %(message)s')
create_logger('error_logger', 'logs/error.log', logging.ERROR, '%(asctime)s %(message)s')
create_logger('training_logger', 'logs/training.log', logging.INFO, '%(asctime)s %(message)s')

def get_logger(name):
    return loggers.get(name)