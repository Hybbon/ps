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


def log_stats_for_folds(rating_set_by_fold):
  logging.info('Ratings for %d folds loaded', len(rating_set_by_fold))
  for fold, rating_set in rating_set_by_fold.items():
    logging.info('Fold %s', fold)
    logging.info('%d base ratings', len(rating_set.base))
    logging.info('%d test ratings', len(rating_set.test))
    if hasattr(rating_set, 'validation'):
      logging.info('%d validation ratings', len(rating_set.validation))
    else:
      logging.info('No validation split')
