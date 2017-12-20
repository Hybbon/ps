import bisect
import collections
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


PercentileSubset = collections.namedtuple('PercentileSubset',
                                          ('subset', 'percentile', 'distance'))


def compute_n_equally_spaced_subsets(sources, distances_frame, size, n):
  logging.info('Computing %d equally-spaced subsets of size %d', n, size)

  distances_and_subsets = []
  for subset in itertools.combinations(sources, size):
    distance_sum = sum_all_pairwise_distances(distances_frame, subset)
    distances_and_subsets.append((distance_sum, subset))

  distances, subsets = zip(*sorted(distances_and_subsets))

  step = max(distances) / (n - 1)
  chosen_subsets = []
  for i in range(n):
    percentile = i / (n - 1) * 100
    percentile_value = i * step
    percentile_index = bisect.bisect_left(distances, percentile_value)

    # bisect_left may return len(distances), if we're unlucky with floating
    # point numbers. In that case, we're on percentile 100% anyway.
    if percentile_index >= len(distances):
      percentile_index = len(distances) - 1

    subset = subsets[percentile_index]
    distance = distances[percentile_index]
    logging.info('Percentile %.2f -- distance = %.6f: %s', percentile, distance,
                 subset)
    chosen_subsets.append(PercentileSubset(list(subset), percentile, distance))

  return chosen_subsets


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
      f'{fold}.{suffix}' for suffix in ['base', 'base.usermap', 'test']
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


class SubsetSymlinker(object):

  def __init__(self, dataset_dir, fold, distances_path):
    self.dataset_dir = dataset_dir
    self.fold = fold

    self.sources = get_source_filenames(dataset_dir, fold)
    logging.info('Sources: %s', self.sources)

    self.distances_frame = load_distances(distances_path)
    logging.info('Distances read successfully from %s', distances_path)

    ensure_no_missing_distances(self.sources, self.distances_frame)

  def compute_subsets_and_make_symlinks(self, output_dir):
    for n in range(2, len(self.sources)):
      size_output_dir = os.path.join(output_dir, f'best-{n}')
      self._compute_subsets_and_make_symlinks_for_size(n, size_output_dir)

  def _compute_subsets_and_make_symlinks_for_size(self, size, output_dir):
    subsets = compute_n_equally_spaced_subsets(self.sources,
                                               self.distances_frame, size, n=7)
    subsets = subsets[1:-1]  # Discard percentiles 0 and 100

    for subset, percentile, distance in subsets:
      logging.info('Got subset at percentile %.2f', percentile)
      percentile_string = f'percentile-{percentile:.2f}'
      percentile_output_dir = os.path.join(output_dir, percentile_string)
      logging.info('Creating symlinks at %s', percentile_output_dir)
      create_symlinks_for_dataset(
          percentile_output_dir,
          self.dataset_dir,
          self.fold,
          source_filenames=subset)
      logging.info('Done creating symlinks.')


def main(dataset_dir, fold, output_dir, distances_path):
  symlinker = SubsetSymlinker(dataset_dir, fold, distances_path)
  symlinker.compute_subsets_and_make_symlinks(output_dir)


def ensure_no_missing_distances(sources, distances_frame):
  missing_sources = set(sources).difference(distances_frame.index)
  if missing_sources:
    raise MissingDistanceDataForSourcesError(missing_sources)


class MissingDistanceDataForSourcesError(Exception):
  pass
