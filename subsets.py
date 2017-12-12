import argparse
import datetime
import logging
import os

from ps import subsets
from ps import logging_utils

def parse_args():
  p = argparse.ArgumentParser(description='Generate ERA distance-optimal input subsets')
  p.add_argument('dataset')
  p.add_argument('--fold', '-f', required=True, choices=['u1', 'u2', 'u3', 'u4', 'u5'])
  p.add_argument('--distances', '-d', required=True, help='path to the distances csv file')
  return p.parse_args()


def main():

  args = parse_args()

  dataset_name = os.path.basename(args.dataset.strip('/'))
  time_string = datetime.datetime.now().strftime('%Y-%m-%d-%Hh%Mm%Ss')
  run_string = 'subsets-{dataset_name}-{fold}-{time}'.format(
      dataset_name=dataset_name, fold=args.fold, time=time_string)

  output_dir = os.path.join('output', run_string)
  log_path = os.path.join('log', run_string, 'run.log')

  for dir_path in [output_dir, os.path.dirname(log_path)]:
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)

  logging_utils.setup_loggers(log_path)

  logging.info('Generating ERA distance-optimal subsets for %s', args.dataset)
  logging.info('Saving stuff at %s', output_dir)
  logging.info('Also logging to %s', log_path)

  subsets.main(dataset_dir=args.dataset, fold=args.fold, output_dir=output_dir, distances_path=args.distances)


if __name__ == '__main__':
  main()
