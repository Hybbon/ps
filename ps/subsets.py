import itertools
import logging
import os

import pandas as pd


def load_distances(csv_path):
  logging.info('Loading distances...')
  distances_frame = pd.read_csv(csv_path, index_col=0)
  logging.info('Distances loaded successfully.')
  return distances_frame


def get_source_filenames(dataset_dir, fold):
  all_files = os.listdir(dataset_dir)
  return [
      filename for filename in all_files
      if filename.startswith(fold) and filename.endswith('.out')
  ]


def compute_distance_optimal_subset(sources, distances_frame, size):
  return max(
      itertools.combinations(sources, size),
      key=lambda subset: sum_all_pairwise_distances(distances_frame, subset))


def sum_all_pairwise_distances(distances_frame, subset):
  subset = list(subset)
  return distances_frame[subset].loc[subset].values.sum()


def create_symlinks_for_dataset(output_dir, dataset_dir, fold,
                                source_filenames):
  reeval_dir = os.path.join(dataset_dir, 'reeval')
  reeval_output_dir = os.path.join(output_dir, 'reeval')

  # We build the list of files to link from the reeval/test dataset because it
  # contains one less file (validation).
  reeval_files_to_link = source_filenames + [
      f'{fold}.{suffix}'
      for suffix in ['base', 'base.usermap', 'test']
  ]

  # The training dataset has the additional validation split
  files_to_link = reeval_files_to_link + [f'{fold}.validation']

  make_split_symlinks(output_dir, dataset_dir, files_to_link)
  make_split_symlinks(reeval_output_dir, reeval_dir, reeval_files_to_link)


def make_split_symlinks(output_dir, dataset_dir, files_to_link):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  for filename in files_to_link:
    # We make these paths absolute because:
    #   1. this makes us not have to think about paths being relative to the
    #      run directory, rather than the output directory
    #   2. this makes the created symlinks work even if we move the created
    #      directory, which should happen more often than moving the dataset
    #      directory.
    source_path = os.path.abspath(os.path.join(dataset_dir, filename))
    destination_path = os.path.abspath(os.path.join(output_dir, filename))
    os.symlink(source_path, destination_path)


def main(dataset_dir, fold, output_dir, distances_path):
  distances_frame = load_distances(distances_path)

  sources = get_source_filenames(dataset_dir, fold)
  logging.info('Sources: %s', sources)

  ensure_no_missing_distances(sources, distances_frame)

  for n in range(2, len(sources)):
    optimal_subset = list(
        compute_distance_optimal_subset(sources, distances_frame, size=n))
    logging.info('Optimal subset for n = %d: %s', n, optimal_subset)

    symlinks_dir = os.path.join(output_dir, f'best-{n}')
    logging.info('Creating symlinks at %s', symlinks_dir)
    create_symlinks_for_dataset(
        symlinks_dir, dataset_dir, fold, source_filenames=optimal_subset)
    logging.info('Done creating symlinks.')


def ensure_no_missing_distances(sources, distances_frame):
  missing_sources = set(sources).difference(distances_frame.index)
  if missing_sources:
    raise MissingDistanceDataForSourcesError(missing_sources)


class MissingDistanceDataForSourcesError(Exception):
  pass
