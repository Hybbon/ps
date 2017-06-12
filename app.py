import argparse
import datetime
import logging
import os

from ps import gen_metrics


def _setup_loggers(log_path):
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


def parse_args():
  p = argparse.ArgumentParser(description='Compute metrics for rankings')
  p.add_argument('dataset')
  return p.parse_args()


def main():

  args = parse_args()

  dataset_name = os.path.basename(args.dataset.strip('/'))
  time_string = datetime.datetime.now().strftime('%Y-%m-%d-%Hh%Mm%Ss')
  run_string = '{dataset_name}-{time}'.format(
      dataset_name=dataset_name, time=time_string)

  output_dir = os.path.join('output', run_string)
  log_path = os.path.join('log', run_string, 'run.log')

  for dir_path in [output_dir, os.path.dirname(log_path)]:
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)

  _setup_loggers(log_path)

  logging.info('Generating metrics for %s', args.dataset)
  logging.info('Saving stuff at %s', output_dir)
  logging.info('Also logging to %s', log_path)

  gen_metrics.main(dataset_dir=args.dataset, output_dir=output_dir)


if __name__ == '__main__':
  main()
