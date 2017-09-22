import unittest

from ps.oracles import map_oracle
from ps import dataset_io

import numpy as np
import pandas as pd

class MAPOracleTest(unittest.TestCase):

  def test_compute_optimal_ranking_set(self):
    ranking_set_u1 = dataset_io.RankingSet(
        id=dataset_io.RankingSetId('u1', 'Alg'),
        matrix=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
        user_ids=[1, 2])

    ranking_set_by_id = {
      ranking_set_u1.id: ranking_set_u1,
    }

    rating_set = dataset_io.RatingSet(fold='u1')
    rating_set.base = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[(4, 1, 5)])
    rating_set.test = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'],
        data=[
            (1, 3, 5),
            (1, 2, 5),
            (2, 1, 5),
            (2, 8, 5),
        ])
    rating_set_by_fold = {'u1': rating_set}

    oracle = map_oracle.MAPOracle(ranking_set_by_id, rating_set_by_fold)

    optimal_ranking_set = oracle.compute_optimal_ranking_set(
      'u1', input_cutoff=4, output_cutoff=3)

    self.assertEqual([1, 2], optimal_ranking_set.user_ids)

    matrix = optimal_ranking_set.matrix
    self.assertEqual(3, len(matrix[0, :]))
    self.assertSetEqual({2, 3}, set(matrix[0, 0:2]))
    self.assertTrue(set(matrix[0, 2:]).issubset({1, 4}))

    self.assertEqual(3, len(matrix[1, :]))
    self.assertSetEqual({8}, set(matrix[1, 0:1]))
    self.assertTrue(set(matrix[1, 1:]).issubset({5, 6, 7}))


if __name__ == '__main__':
  unittest.main()
