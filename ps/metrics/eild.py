import collections
import logging
import math

import numpy as np
import pandas as pd


def _compute_likers(ratings):
  return {
      item_id: set(item_frame.user_id)
      for item_id, item_frame in ratings.groupby(['item_id'])
  }


def _compute_likers_by_fold(rating_set_by_fold):
  likers_by_fold = {
      fold: _compute_likers(rating_set.all_ratings())
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


class EILD(object):
  NAME = 'EILD'

  def __init__(self, ranking_set_by_id, rating_set_by_fold):
    logging.info('Computing likers')
    self.likers_by_fold = _compute_likers_by_fold(rating_set_by_fold)
    logging.info('Done computing likers')

  def compute(self, ranking_set, num_items=None):
    likers_by_item = self.likers_by_fold[ranking_set.id.fold]

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

        distances = _compute_distances(k_items, l_items, likers_by_item)
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
