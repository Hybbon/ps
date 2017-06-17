import collections
import re
import os

import numpy as np
import pandas as pd

_RANKING_SET_FILENAME_REGEX = r'u(\d+)-(\w+)\.out'

RankingSetId = collections.namedtuple('RankingSetId', ('fold', 'source'))


def _yield_ranking_set_filenames(dir_path):
  for filename in os.listdir(dir_path):
    match = re.match(_RANKING_SET_FILENAME_REGEX, filename)
    if match:
      fold, source = match.groups()
      yield filename, RankingSetId(fold, source)


_RATING_SET_FILENAME_REGEX = r'u(\d+)\.(base|test|validation)'


def _yield_rating_set_filenames(dir_path):
  for filename in os.listdir(dir_path):
    match = re.match(_RATING_SET_FILENAME_REGEX, filename)
    if match:
      fold, split = match.groups()
      yield filename, fold, split


def _parse_ranking_line(line):
  user_id_string, ranking_string = line.split('\t')

  position_strings = ranking_string[1:-1].split(',')
  item_ids_and_scores = (position_string.split(':')
                         for position_string in position_strings)
  item_ids = [int(item_id) for item_id, _ in item_ids_and_scores]

  return int(user_id_string), item_ids


RankingSet = collections.namedtuple('RankingSet', ('id', 'matrix', 'user_ids'))


def _load_ranking_matrix(path):
  user_ids = []
  rankings = []

  with open(path) as f:
    for line in f:
      user_id, ranking = _parse_ranking_line(line)
      user_ids.append(user_id)
      rankings.append(ranking)

  max_ranking_length = max(len(ranking) for ranking in rankings)

  rankings_matrix = np.full(
      (len(rankings), max_ranking_length), -1, dtype='int32')
  for i, ranking in enumerate(rankings):
    rankings_matrix[i, :len(ranking)] = ranking

  return rankings_matrix, user_ids


def load_ranking_sets(dir_path):
  ranking_set_by_id = {}
  for filename, ranking_set_id in _yield_ranking_set_filenames(dir_path):
    ranking_matrix, user_ids = _load_ranking_matrix(
        os.path.join(dir_path, filename))
    ranking_set_by_id[ranking_set_id] = RankingSet(ranking_set_id,
                                                   ranking_matrix, user_ids)

  sorted_ids_and_ranking_sets = sorted(
      ranking_set_by_id.items(), key=lambda r: r[0])

  return collections.OrderedDict(sorted_ids_and_ranking_sets)


def _load_ratings(path):
  column_names = ('user_id', 'item_id', 'rating')
  return pd.read_csv(
      path,
      "\t",
      names=column_names)


class RatingSet(object):

  def __init__(self, fold):
    self.fold = fold

  def all_ratings(self):
    frame = self.base.append(self.test, ignore_index=True)
    if hasattr(self, 'validation'):
      frame = frame.append(self.validation, ignore_index=True)

    return frame


def load_ratings_for_all_folds(dir_path):
  rating_set_by_fold = {}
  for filename, fold, split in _yield_rating_set_filenames(dir_path):
    ratings = _load_ratings(os.path.join(dir_path, filename))
    rating_set = rating_set_by_fold.get(fold)
    if rating_set is None:
      rating_set = RatingSet(fold)
      rating_set_by_fold[fold] = rating_set
    setattr(rating_set, split, ratings)

  return collections.OrderedDict(sorted(rating_set_by_fold.items()))
