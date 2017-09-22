import math
import unittest

from ps.metrics import eild
from ps import dataset_io

import numpy as np
import pandas as pd


class EildComputeTest(unittest.TestCase):

  def setUp(self):
    super().setUp()

    rating_set = dataset_io.RatingSet(fold='u1')
    rating_set.base = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[])
    rating_set.test = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[])
    rating_set_by_fold = {'u1': rating_set}

    self.eild = eild.EILD(
        ranking_set_by_id={}, rating_set_by_fold=rating_set_by_fold)

  def test_ranking_with_single_item(self):
    ranking_matrix = np.array([[1]])
    ranking_set = dataset_io.RankingSet(
        id=dataset_io.RankingSetId('u1', 'Alg'),
        matrix=ranking_matrix,
        user_ids=[])

    self.assertAlmostEqual(0, self.eild.compute(ranking_set))

  def test_ranking_with_item_not_in_ratings(self):
    item_ids = [1]
    distance_matrix = np.array([
        [1]
    ])
    self.eild.distances_by_fold = {'u1': (distance_matrix, item_ids)}
    ranking_matrix = np.array([[1, 2]])
    ranking_set = dataset_io.RankingSet(
        id=dataset_io.RankingSetId('u1', 'Alg'),
        matrix=ranking_matrix,
        user_ids=[])

    self.assertAlmostEqual(0, self.eild.compute(ranking_set))

  def test_many_items(self):
    item_ids = [1, 2, 3]
    distance_matrix = np.array([
        [1, 2 / math.sqrt(6), 1 / math.sqrt(3)],
        [2 / math.sqrt(6), 1, 1 / math.sqrt(2)],
        [1 / math.sqrt(3), 1 / math.sqrt(2), 1],
    ])
    self.eild.distances_by_fold = {'u1': (distance_matrix, item_ids)}

    ranking_matrix = np.array([[1, 2, 3]])
    ranking_set = dataset_io.RankingSet(
        id=dataset_io.RankingSetId('u1', 'Alg'),
        matrix=ranking_matrix,
        user_ids=[])

    eild_k0 = (2 / math.sqrt(6) + 0.85 / math.sqrt(3)) / 1.85
    eild_k1 = (2 / math.sqrt(6) + 1 / math.sqrt(2)) / 2
    eild_k2 = (1 / math.sqrt(3) + 1 / math.sqrt(2)) / 2

    expected_eild = (eild_k0 + eild_k1 * 0.85 + eild_k2 * 0.85**2) / (
        1 + 0.85 + 0.85**2)

    self.assertAlmostEqual(expected_eild, self.eild.compute(ranking_set))

  def test_returns_mean_for_many_rankings(self):
    item_ids = [1, 2, 3]
    distance_matrix = np.array([
        [1, 2 / math.sqrt(6), 1 / math.sqrt(3)],
        [2 / math.sqrt(6), 1, 1 / math.sqrt(2)],
        [1 / math.sqrt(3), 1 / math.sqrt(2), 1],
    ])
    self.eild.distances_by_fold = {'u1': (distance_matrix, item_ids)}

    ranking_matrix = np.array([[1, 2], [2, 3]])
    ranking_set = dataset_io.RankingSet(
        id=dataset_io.RankingSetId('u1', 'Alg'),
        matrix=ranking_matrix,
        user_ids=[])

    eild_k0_u0 = eild_k1_u0 = 2 / math.sqrt(6)
    eild_u0 = (eild_k0_u0 + eild_k1_u0 * 0.85) / (1 + 0.85)

    eild_k0_u1 = eild_k1_u1 = 1 / math.sqrt(2)
    eild_u1 = (eild_k0_u1 + eild_k1_u1 * 0.85) / (1 + 0.85)

    expected_eild = (eild_u0 + eild_u1) / 2

    self.assertAlmostEqual(expected_eild, self.eild.compute(ranking_set))


if __name__ == '__main__':
  unittest.main()
