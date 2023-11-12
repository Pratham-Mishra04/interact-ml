import logging

logging.basicConfig(filename="logs/info.log", level=logging.INFO, format='%(asctime)s %(message)s', filemode='w')
info_logger = logging.getLogger('info_logger')

logging.basicConfig(filename="logs/warn.log", level=logging.WARNING, format='%(asctime)s %(message)s', filemode='w')
warn_logger = logging.getLogger('warn_logger')

logging.basicConfig(filename="logs/error.log", level=logging.ERROR, format='%(asctime)s %(message)s', filemode='w')
error_logger = logging.getLogger('error_logger')