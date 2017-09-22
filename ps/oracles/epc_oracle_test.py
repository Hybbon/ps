import unittest

from ps.oracles import epc_oracle
from ps import dataset_io

import numpy as np
import pandas as pd

class EPCOracleTest(unittest.TestCase):

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
        columns=['user_id', 'item_id', 'rating'], data=[])

    rating_set_by_fold = {'u1': rating_set}

    oracle = epc_oracle.EPCOracle(ranking_set_by_id, rating_set_by_fold)

    oracle.popularity_by_fold = {'u1': {
      1: 1., 2: 0.25, 3: 0.5, 4: 0.75,
      5: 0.8, 6: 0.3, 7: 0.6, 8: 0.01
    }}

    optimal_ranking_set = oracle.compute_optimal_ranking_set(
      'u1', input_cutoff=4, output_cutoff=3)

    self.assertEqual([1, 2], optimal_ranking_set.user_ids)

    matrix = optimal_ranking_set.matrix
    self.assertEqual([2, 3, 4], list(matrix[0, :]))
    self.assertEqual([8, 6, 7], list(matrix[1, :]))



if __name__ == '__main__':
  unittest.main()
