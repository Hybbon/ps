import numpy as np

from ps import dataset_io
from ps import rating_utils
from ps.oracles import oracle_utils

class MAPOracle(oracle_utils.Oracle):

  def __init__(self, ranking_set_by_id, rating_set_by_fold):
    super().__init__(ranking_set_by_id, rating_set_by_fold)
    self.hits_by_fold = rating_utils.compute_hits_by_fold(rating_set_by_fold)

  def compute_optimal_ranking_set(self, fold, input_cutoff, output_cutoff):
    hits_by_user = self.hits_by_fold[fold]
    recommended_to_user = self._compute_recommended_in_fold(fold, input_cutoff)

    matrix = np.ndarray((len(recommended_to_user), output_cutoff))
    user_ids = []

    for user_index, (user_id, recommended_items) in enumerate(recommended_to_user.items()):
      user_ids.append(user_id)

      rank = 0
      for item_id in recommended_items:
        if item_id in hits_by_user[user_id]:
          matrix[user_index, rank] = item_id
          rank += 1
          if rank == output_cutoff:
            break
      else:
        for item_id in recommended_items:
          if item_id not in hits_by_user[user_id]:
            matrix[user_index, rank] = item_id
            rank += 1
            if rank == output_cutoff:
              break

    ranking_set_id = dataset_io.RankingSetId(fold, 'MAPOracle')
    ranking_set = dataset_io.RankingSet(ranking_set_id, matrix, user_ids)
    return ranking_set

