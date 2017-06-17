import collections
import logging
import math

import numpy as np
import pandas as pd


def _compute_likers(ratings):
  return {
      item_id: frozenset(item_frame.user_id)
      for item_id, item_frame in ratings.groupby(['item_id'])
  }


def _compute_likers_by_fold(rating_set_by_fold):
  likers_by_fold = {
      fold: _compute_likers(rating_set.base)
      for fold, rating_set in rating_set_by_fold.items()
  }
  return collections.OrderedDict(sorted(likers_by_fold.items()))


def _cosine_similarity(a, b):
  return len(a & b) / ((math.sqrt(len(a)) * math.sqrt(len(b))) or 1)


def _compute_distances(k_items, l_items, likers_by_item):
  distances = []
  for k_item, l_item in zip(k_items, l_items):
    k_likers = likers_by_item.get(k_item, set())
    l_likers = likers_by_item.get(l_item, set())
    distance = _cosine_similarity(k_likers, l_likers)
    distances.append(distance)
  return distances


def _compute_distance_matrix(ratings):
  likers_by_item = _compute_likers(ratings)
  item_ids = sorted(likers_by_item.keys())
  num_items = len(item_ids)
  matrix = np.zeros((num_items, num_items))
  for i, item_i in enumerate(item_ids):
    for j, item_j in enumerate(item_ids):
      i_likers = likers_by_item.get(item_i, frozenset())
      j_likers = likers_by_item.get(item_j, frozenset())
      matrix[i][j] = _cosine_similarity(i_likers, j_likers)

  return matrix, item_ids


def _compute_distances_by_fold(rating_set_by_fold):
  likers_by_fold = {
      fold: _compute_distance_matrix(rating_set.base)
      for fold, rating_set in rating_set_by_fold.items()
  }
  return collections.OrderedDict(sorted(likers_by_fold.items()))


class EILD(object):
  NAME = 'EILD'

  def __init__(self, ranking_set_by_id, rating_set_by_fold):
    logging.info('Computing distances')
    self.distances_by_fold = _compute_distances_by_fold(rating_set_by_fold)
    logging.info('Done computing distances')

  def compute(self, ranking_set, num_items=None):
    distance_matrix, item_ids = self.distances_by_fold[ranking_set.id.fold]
    index_by_item_id = {item_id: i for i, item_id in enumerate(item_ids)}

    matrix = ranking_set.matrix

    if num_items is not None:
      matrix = matrix[:, :num_items]
    else:
      num_items = matrix.shape[1]

    total_eild = 0
    normalizing_constant = 0

    for k in range(num_items):
      k_items = matrix[:, k]
      k_eild = 0
      k_normalizing_constant = 0

      for l in range(num_items):
        if k == l:
          continue

        l_items = matrix[:, l]

        distances = []
        for k_item, l_item in zip(k_items, l_items):
          k_index = index_by_item_id[k_item]
          l_index = index_by_item_id[l_item]

          # If an item appears in the rankings, but not in any rating.
          if k_index is None or l_index is None:
            distances.append(0.)
          else:
            distances.append(distance_matrix[k_index, l_index])

        mean_distance = sum(distances) / len(distances)

        relative_discount = 0.85**max(0, l - k - 1)
        k_eild += mean_distance * relative_discount
        k_normalizing_constant += relative_discount

      if k_normalizing_constant != 0:
        k_eild /= k_normalizing_constant

      absolute_discount = 0.85**k
      total_eild += absolute_discount * k_eild

      normalizing_constant += absolute_discount

    if normalizing_constant != 0:
      total_eild /= normalizing_constant

    return total_eild
