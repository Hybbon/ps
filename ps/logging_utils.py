import logging

def setup_loggers(log_path):

  logging_formatter = logging.Formatter(
      "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
  root_logger = logging.getLogger()
  root_logger.setLevel(logging.DEBUG)

  file_handler = logging.FileHandler(log_path)
  file_handler.setFormatter(logging_formatter)
  root_logger.addHandler(file_handler)

  console_handler = logging.StreamHandler()
  console_handler.setFormatter(logging_formatter)
  root_logger.addHandler(console_handler)
