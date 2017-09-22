import unittest

from ps.metrics import map as map_module
from ps import dataset_io

import numpy as np
import pandas as pd


class MapComputeTest(unittest.TestCase):

  def setUp(self):
    super().setUp()

    rating_set = dataset_io.RatingSet(fold='u1')
    rating_set.base = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[])
    rating_set.test = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[])
    rating_set_by_fold = {'u1': rating_set}

    self.map = map_module.MAP(
        ranking_set_by_id={}, rating_set_by_fold=rating_set_by_fold)

  def test_ranking_with_single_hit(self):
    hits_by_user = {1: {1}}
    self.map.hits_by_fold = {'u1': hits_by_user}

    ranking_matrix = np.array([[1]])
    ranking_set = dataset_io.RankingSet(
        id=dataset_io.RankingSetId('u1', 'Alg'),
        matrix=ranking_matrix,
        user_ids=[1])

    self.assertAlmostEqual(1, self.map.compute(ranking_set))

  def test_ranking_with_single_miss(self):
    hits_by_user = {1: {1}}
    self.map.hits_by_fold = {'u1': hits_by_user}

    ranking_matrix = np.array([[1]])
    ranking_set = dataset_io.RankingSet(
        id=dataset_io.RankingSetId('u1', 'Alg'),
        matrix=ranking_matrix,
        user_ids=[1])

    self.assertAlmostEqual(1, self.map.compute(ranking_set))

  def test_complex_ranking_with_all_hits(self):
    hits_by_user = {1: {1, 3}}
    self.map.hits_by_fold = {'u1': hits_by_user}

    ranking_matrix = np.array([[1, 2, 3]])
    ranking_set = dataset_io.RankingSet(
        id=dataset_io.RankingSetId('u1', 'Alg'),
        matrix=ranking_matrix,
        user_ids=[1])

    expected_map = (1 + 2 / 3) / 2

    self.assertAlmostEqual(expected_map, self.map.compute(ranking_set))

  def test_complex_ranking_with_missing_hits(self):
    hits_by_user = {1: {1, 3, 4}}
    self.map.hits_by_fold = {'u1': hits_by_user}

    ranking_matrix = np.array([[1, 2, 5, 3]])
    ranking_set = dataset_io.RankingSet(
        id=dataset_io.RankingSetId('u1', 'Alg'),
        matrix=ranking_matrix,
        user_ids=[1])

    expected_map = (1 + 2 / 4) / 3

    self.assertAlmostEqual(expected_map, self.map.compute(ranking_set))

  def test_returns_mean_for_many_rankings(self):
    hits_by_user = {1: {1}, 2: {}}
    self.map.hits_by_fold = {'u1': hits_by_user}

    ranking_matrix = np.array([[1], [1]])
    ranking_set = dataset_io.RankingSet(
        id=dataset_io.RankingSetId('u1', 'Alg'),
        matrix=ranking_matrix,
        user_ids=[1, 2])

    self.assertAlmostEqual(0.5, self.map.compute(ranking_set))


if __name__ == '__main__':
  unittest.main()
