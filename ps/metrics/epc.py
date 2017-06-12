import collections
import logging

import numpy as np
import pandas as pd


def _compute_popularity(ratings):
  """Computes the popularity of the items in the ratings.

  ratings -- A DataFrame with all ratings.

  Returns a Series of the popularity indexed by item_id. Popularity values
  range from 0 to 1.
  """
  num_users = len(ratings.user_id.unique()) or 1  # No ratings.
  return ratings.item_id.value_counts() / num_users


def _compute_popularity_by_fold(rating_set_by_fold):
  popularity_by_fold = {
      fold: _compute_popularity(rating_set.all_ratings())
      for fold, rating_set in rating_set_by_fold.items()
  }
  return collections.OrderedDict(sorted(popularity_by_fold.items()))


class EPC(object):
  NAME = 'EPC'

  def __init__(self, ranking_set_by_id, rating_set_by_fold):
    logging.info('Computing popularity')
    self.popularity_by_fold = _compute_popularity_by_fold(rating_set_by_fold)
    logging.info('Done computing popularity')

  def compute(self, ranking_set, num_items=None):
    item_popularity = self.popularity_by_fold[ranking_set.id.fold]

    matrix = ranking_set.matrix

    if num_items is not None:
      matrix = matrix[:, :num_items]
    else:
      num_items = matrix.shape[1]

    discount = 0.85**np.arange(num_items)

    popularity_for_item = np.vectorize(
        lambda item_id: item_popularity.get(item_id, 0.), otypes=[np.float])
    popularity_matrix = popularity_for_item(matrix)

    epcs_for_rankings = (1 - popularity_matrix).dot(discount) / discount.sum()

    return epcs_for_rankings.mean()
