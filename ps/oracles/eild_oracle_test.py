import unittest

from ps.oracles import eild_oracle
from ps import dataset_io

import numpy as np
import pandas as pd

class EILDOracleTest(unittest.TestCase):

  def test_compute_optimal_ranking_set(self):
    ranking_set_u1 = dataset_io.RankingSet(
        id=dataset_io.RankingSetId('u1', 'Alg'),
        matrix=np.array([[1, 2, 3]]),
        user_ids=[1])

    ranking_set_by_id = {
      ranking_set_u1.id: ranking_set_u1,
    }

    rating_set = dataset_io.RatingSet(fold='u1')
    rating_set.base = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[])

    rating_set_by_fold = {'u1': rating_set}

    oracle = eild_oracle.EILDOracle(ranking_set_by_id, rating_set_by_fold)

    distance_matrix = np.array(
      [[0, 0.5, 0.2],
       [0.5, 0, 0.9],
       [0.2, 0.9, 0]])
    oracle.distances_by_fold = {'u1': (distance_matrix, [1, 2, 3])}

    optimal_ranking_set = oracle.compute_optimal_ranking_set(
      'u1', input_cutoff=3, output_cutoff=2)

    self.assertEqual([1], optimal_ranking_set.user_ids)

    matrix = optimal_ranking_set.matrix
    self.assertEqual([2, 3], list(matrix[0, :]))



if __name__ == '__main__':
  unittest.main()
