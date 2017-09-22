import collections
import logging
import math

import pandas as pd

from ps import rating_utils


class EILD(object):
  NAME = 'EILD'

  def __init__(self, ranking_set_by_id, rating_set_by_fold):
    logging.info('Computing distances')
    self.distances_by_fold = rating_utils.compute_distances_by_fold(rating_set_by_fold)
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
          k_index = index_by_item_id.get(k_item)
          l_index = index_by_item_id.get(l_item)

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
