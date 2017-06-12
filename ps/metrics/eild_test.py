import math
import unittest

from ps.metrics import eild
from ps import dataset_io

import numpy as np
import pandas as pd


class ComputeLikersTest(unittest.TestCase):

  def test_rating_set_with_validation(self):
    rating_set = dataset_io.RatingSet(fold='u1')
    rating_set.base = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[(1, 2, 5), (2, 2, 5)])
    rating_set.test = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[(2, 3, 5)])
    rating_set.validation = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[(3, 3, 5)])
    rating_set_by_fold = {'u1': rating_set}

    likers_by_fold = eild._compute_likers_by_fold(rating_set_by_fold)
    self.assertIn('u1', likers_by_fold)

    likers_by_item_id = likers_by_fold['u1']
    self.assertDictEqual({2: {1, 2}, 3: {2, 3}}, likers_by_item_id)

  def test_rating_set_without_validation(self):
    rating_set = dataset_io.RatingSet(fold='u1')
    rating_set.base = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[(1, 2, 5), (2, 2, 5)])
    rating_set.test = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[(2, 3, 5)])
    rating_set_by_fold = {'u1': rating_set}

    likers_by_fold = eild._compute_likers_by_fold(rating_set_by_fold)
    self.assertIn('u1', likers_by_fold)

    likers_by_item_id = likers_by_fold['u1']
    self.assertDictEqual({2: {1, 2}, 3: {2}}, likers_by_item_id)


class ComputeDistanceMatrixTest(unittest.TestCase):

  def test_rating_set_with_validation(self):
    rating_set = dataset_io.RatingSet(fold='u1')
    rating_set.base = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[(1, 2, 5), (2, 2, 5)])
    rating_set.test = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[(2, 3, 5)])
    rating_set.validation = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[(3, 3, 5)])
    rating_set_by_fold = {'u1': rating_set}

    distance_by_fold = eild._compute_distances_by_fold(rating_set_by_fold)
    self.assertIn('u1', distance_by_fold)

    distance_matrix, item_ids = distance_by_fold['u1']
    expected_matrix = np.array([[1., .5], [.5, 1.]])

    are_almost_equal = ((distance_matrix - expected_matrix) < 1e-8).all()
    self.assertTrue(are_almost_equal)

    self.assertSequenceEqual([2, 3], item_ids)

  def test_rating_set_without_validation(self):
    rating_set = dataset_io.RatingSet(fold='u1')
    rating_set.base = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[(1, 2, 5), (2, 2, 5)])
    rating_set.test = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[(2, 3, 5)])
    rating_set_by_fold = {'u1': rating_set}

    distance_by_fold = eild._compute_distances_by_fold(rating_set_by_fold)
    self.assertIn('u1', distance_by_fold)

    distance_matrix, item_ids = distance_by_fold['u1']
    expected_matrix = np.array([[1., 1 / math.sqrt(2)], [1 / math.sqrt(2), 1.]])

    are_almost_equal = ((distance_matrix - expected_matrix) < 1e-8).all()
    self.assertTrue(are_almost_equal)

    self.assertSequenceEqual([2, 3], item_ids)


class CosineSimilarityTest(unittest.TestCase):

  def test_cosine_similarity_both_empty(self):
    value = eild._cosine_similarity(set(), set())
    self.assertAlmostEqual(0, value)

  def test_cosine_similarity_one_empty(self):
    value = eild._cosine_similarity({1}, set())
    self.assertAlmostEqual(0, value)

  def test_cosine_similarity_disjoint_sets(self):
    value = eild._cosine_similarity({1}, {2, 3})
    self.assertAlmostEqual(0, value)

  def test_cosine_similarity_equal_sets(self):
    value = eild._cosine_similarity({2, 3}, {2, 3})
    self.assertAlmostEqual(1, value)

  def test_cosine_similarity_intersection(self):
    value = eild._cosine_similarity({1, 2}, {2, 3})
    self.assertAlmostEqual(0.5, value)

  def test_cosine_similarity_subset(self):
    value = eild._cosine_similarity({1, 2}, {1})
    self.assertAlmostEqual(1 / math.sqrt(2), value)


class ComputeDistancesTest(unittest.TestCase):

  def test_no_likers(self):
    likers_per_item = {}
    distances = eild._compute_distances([1, 3], [2, 4], likers_per_item)
    self.assertSequenceEqual([0, 0], distances)

  def test_with_likers(self):
    likers_per_item = {1: {2, 3}, 2: {2, 3}, 3: {2}, 4: {2, 3}}
    distances = eild._compute_distances([1, 3], [2, 4], likers_per_item)

    self.assertEqual(2, len(distances))
    self.assertAlmostEqual(1, distances[0])
    self.assertAlmostEqual(1 / math.sqrt(2), distances[1])


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
