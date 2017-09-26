import functools
import heapq

import numpy as np
import pandas as pd

from ps import dataset_io
from ps import rating_utils
from ps.oracles import oracle_utils


class EILDOracle(oracle_utils.Oracle):

  def __init__(self, ranking_set_by_id, rating_set_by_fold):
    super().__init__(ranking_set_by_id, rating_set_by_fold)
    self.distances_by_fold = rating_utils.compute_distances_by_fold(rating_set_by_fold)

  def compute_optimal_ranking_set(self, fold, input_cutoff, output_cutoff):
    distance_matrix, item_ids = self.distances_by_fold[fold]

    mean_distance_by_item = pd.Series(distance_matrix.sum(axis=0), index=item_ids)

    recommended_to_user = self._compute_recommended_in_fold(fold, input_cutoff)

    matrix = np.ndarray((len(recommended_to_user), output_cutoff))
    user_ids = []

    for i, (user_id, recommended_to_user) in enumerate(recommended_to_user.items()):
      user_ids.append(user_id)

      matrix[i, :] = heapq.nlargest(output_cutoff, recommended_to_user,
              lambda i: mean_distance_by_item.get(i, 0.))

    ranking_set_id = dataset_io.RankingSetId(fold, 'EILDOracle')
    ranking_set = dataset_io.RankingSet(ranking_set_id, matrix, user_ids)
    return ranking_set
