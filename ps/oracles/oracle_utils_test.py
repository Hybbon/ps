import unittest

from ps.oracles import oracle_utils
from ps import dataset_io

import numpy as np

class OracleTest(unittest.TestCase):

  def test_compute_recommended_in_fold(self):
    ranking_set_u1 = dataset_io.RankingSet(
        id=dataset_io.RankingSetId('u1', 'Alg'),
        matrix=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
        user_ids=[1, 2])

    ranking_set_u2 = dataset_io.RankingSet(
        id=dataset_io.RankingSetId('u2', 'Alg'),
        matrix=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
        user_ids=[2, 3])

    ranking_set_by_id = {
      ranking_set_u1.id: ranking_set_u1,
      ranking_set_u2.id: ranking_set_u2,
    }

    oracle = oracle_utils.Oracle(
        ranking_set_by_id, rating_set_by_fold={})

    recommended_in_fold = oracle._compute_recommended_in_fold('u1', cutoff=3)

    expected = {1: {1, 2, 3}, 2: {5, 6, 7}}

    self.assertDictEqual(expected, recommended_in_fold)


if __name__ == '__main__':
  unittest.main()
