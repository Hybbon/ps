import argparse
import datetime
import logging
import os

from ps import gen_oracles
from ps import logging_utils

def parse_args():
  p = argparse.ArgumentParser(description='Generate oracle rankings')
  p.add_argument('dataset')
  return p.parse_args()


def main():

  args = parse_args()

  dataset_name = os.path.basename(args.dataset.strip('/'))
  time_string = datetime.datetime.now().strftime('%Y-%m-%d-%Hh%Mm%Ss')
  run_string = 'oracle-{dataset_name}-{time}'.format(
      dataset_name=dataset_name, time=time_string)

  output_dir = os.path.join('output', run_string)
  log_path = os.path.join('log', run_string, 'run.log')

  for dir_path in [output_dir, os.path.dirname(log_path)]:
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)

  logging_utils.setup_loggers(log_path)

  logging.info('Generating oracles for %s', args.dataset)
  logging.info('Saving stuff at %s', output_dir)
  logging.info('Also logging to %s', log_path)

  gen_oracles.main(dataset_dir=args.dataset, output_dir=output_dir)


if __name__ == '__main__':
  main()
