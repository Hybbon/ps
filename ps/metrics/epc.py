import collections
import logging

import numpy as np
import pandas as pd

from ps import rating_utils


class EPC(object):
  NAME = 'EPC'

  def __init__(self, ranking_set_by_id, rating_set_by_fold):
    logging.info('Computing popularity')
    self.popularity_by_fold = rating_utils.compute_popularity_by_fold(rating_set_by_fold)
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
