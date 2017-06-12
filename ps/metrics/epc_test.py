import unittest

from ps.metrics import epc
from ps import dataset_io

import numpy as np
import pandas as pd


class ComputePopularityTest(unittest.TestCase):

  def test_compute_popularity(self):
    ratings = pd.DataFrame.from_records(
        [
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 3),
        ], columns=['user_id', 'item_id'])

    expected_popularity = {(1, 1.), (2, 0.5), (3, 0.5)}
    popularity = set(epc._compute_popularity(ratings).items())

    self.assertEqual(expected_popularity, popularity)


class EpcComputeTest(unittest.TestCase):

  def setUp(self):
    super().setUp()

    rating_set = dataset_io.RatingSet(fold='u1')
    rating_set.base = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[])
    rating_set.test = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[])
    rating_set_by_fold = {'u1': rating_set}

    self.epc = epc.EPC(
        ranking_set_by_id={}, rating_set_by_fold=rating_set_by_fold)

  def test_single_item(self):
    ranking_matrix = np.array([[1]])
    ranking_set = dataset_io.RankingSet(
        id=dataset_io.RankingSetId('u1', 'Alg'),
        matrix=ranking_matrix,
        user_ids=[])

    self.epc.popularity_by_fold = {'u1': pd.Series({1: 1})}

    value = self.epc.compute(ranking_set)

    self.assertAlmostEqual(0, value)

  def test_many_items(self):
    ranking_matrix = np.array([[1, 2, 3]])
    ranking_set = dataset_io.RankingSet(
        id=dataset_io.RankingSetId('u1', 'Alg'),
        matrix=ranking_matrix,
        user_ids=[])

    self.epc.popularity_by_fold = {'u1': pd.Series({1: 0.5, 2: 0.3, 3: 0.1})}

    value = self.epc.compute(ranking_set)

    expected_value = (0.5 + 0.7 * 0.85 + 0.9 * 0.85**2) / (1 + 0.85 + 0.85**2)
    self.assertAlmostEqual(expected_value, value)

  def test_returns_mean_for_many_rankings(self):
    ranking_matrix = np.array([[1], [2]])
    ranking_set = dataset_io.RankingSet(
        id=dataset_io.RankingSetId('u1', 'Alg'),
        matrix=ranking_matrix,
        user_ids=[])

    self.epc.popularity_by_fold = {'u1': pd.Series({1: 0.5, 2: 0.3})}

    value = self.epc.compute(ranking_set)

    expected_value = (0.5 + 0.7) / 2
    self.assertAlmostEqual(expected_value, value)


if __name__ == '__main__':
  unittest.main()
