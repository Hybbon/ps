import math
import unittest

import numpy as np
import pandas as pd

from ps import dataset_io
from ps import rating_utils

class ComputeHitsTest(unittest.TestCase):

  def test_compute_hits_by_fold(self):
    rating_set = dataset_io.RatingSet(fold='u1')
    rating_set.base = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[(4, 1, 5)])
    rating_set.test = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'],
        data=[
            (1, 1, 5),
            (1, 2, 5),
            (2, 1, 5),
            (2, 3, 5),
        ])
    rating_set_by_fold = {'u1': rating_set}

    hits_by_fold = rating_utils.compute_hits_by_fold(rating_set_by_fold)
    self.assertIn('u1', hits_by_fold)

    hits = hits_by_fold['u1']

    expected_hits = {1: {1, 2}, 2: {1, 3}}

    self.assertDictEqual(expected_hits, hits)

  def test_compute_rated_by_fold(self):
    rating_set = dataset_io.RatingSet(fold='u1')
    rating_set.base = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'],
        data=[
            (1, 1, 5),
            (1, 2, 5),
            (2, 1, 5),
            (2, 3, 5),
        ])
    rating_set_by_fold = {'u1': rating_set}

    rated_by_fold = rating_utils.compute_rated_by_fold(
        rating_set_by_fold, split_name='base')
    self.assertIn('u1', rated_by_fold)

    rated = rated_by_fold['u1']

    expected_rated = {1: {1, 2}, 2: {1, 3}}

    self.assertDictEqual(expected_rated, rated)


class ComputePopularityTest(unittest.TestCase):

  def test_compute_popularity(self):
    rating_set = dataset_io.RatingSet(fold='u1')
    rating_set.base = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'],
        data=[
            (1, 1, 5),
            (1, 2, 5),
            (2, 1, 5),
            (2, 3, 5),
        ])
    rating_set_by_fold = {'u1': rating_set}

    expected_popularity = {(1, 1.), (2, 0.5), (3, 0.5)}
    popularity = set(rating_utils._compute_popularity(rating_set).items())

    self.assertEqual(expected_popularity, popularity)


class ComputeLikersTest(unittest.TestCase):

  def test_computes_likers(self):
    rating_set = dataset_io.RatingSet(fold='u1')
    rating_set.base = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'],
        data=[(1, 2, 5), (2, 2, 5), (2, 3, 5), (3, 3, 5)])
    rating_set_by_fold = {'u1': rating_set}

    likers_by_fold = rating_utils._compute_likers_by_fold(rating_set_by_fold)
    self.assertIn('u1', likers_by_fold)

    likers_by_item_id = likers_by_fold['u1']
    self.assertDictEqual({2: {1, 2}, 3: {2, 3}}, likers_by_item_id)

  def test_ignores_test_and_validation(self):
    rating_set = dataset_io.RatingSet(fold='u1')
    rating_set.base = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[])
    rating_set.test = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[(2, 3, 5)])
    rating_set.validation = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[(1, 3, 5)])
    rating_set_by_fold = {'u1': rating_set}

    likers_by_fold = rating_utils._compute_likers_by_fold(rating_set_by_fold)
    self.assertIn('u1', likers_by_fold)

    likers_by_item_id = likers_by_fold['u1']
    self.assertDictEqual({}, likers_by_item_id)


class ComputeDistanceMatrixTest(unittest.TestCase):

  def test_computes_distances(self):
    rating_set = dataset_io.RatingSet(fold='u1')
    rating_set.base = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'],
        data=[(1, 2, 5), (2, 2, 5), (4, 2, 5), (2, 3, 5), (3, 3, 5), (4, 3, 5)])
    rating_set_by_fold = {'u1': rating_set}

    distance_by_fold = rating_utils.compute_distances_by_fold(rating_set_by_fold)
    self.assertIn('u1', distance_by_fold)

    distance_matrix, item_ids = distance_by_fold['u1']

    expected_matrix = np.array([[0., 1/3], [1/3, 0.]])
    error_matrix = np.abs(distance_matrix - expected_matrix)
    are_almost_equal = (error_matrix < 1e-8).all()

    self.assertTrue(are_almost_equal)

    self.assertSequenceEqual([2, 3], item_ids)

  def test_ignores_test_and_validation(self):
    rating_set = dataset_io.RatingSet(fold='u1')
    rating_set.base = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[])
    rating_set.test = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[(2, 3, 5)])
    rating_set.validation = pd.DataFrame.from_records(
        columns=['user_id', 'item_id', 'rating'], data=[(1, 3, 5)])
    rating_set_by_fold = {'u1': rating_set}

    distance_by_fold = rating_utils.compute_distances_by_fold(rating_set_by_fold)
    self.assertIn('u1', distance_by_fold)

    distance_matrix, item_ids = distance_by_fold['u1']
    expected_matrix = np.array([])

    are_almost_equal = ((distance_matrix - expected_matrix) < 1e-8).all()
    self.assertTrue(are_almost_equal)

    self.assertSequenceEqual([], item_ids)


class CosineSimilarityTest(unittest.TestCase):

  def test_cosine_similarity_both_empty(self):
    value = rating_utils._cosine_similarity(set(), set())
    self.assertAlmostEqual(0, value)

  def test_cosine_similarity_one_empty(self):
    value = rating_utils._cosine_similarity({1}, set())
    self.assertAlmostEqual(0, value)

  def test_cosine_similarity_disjoint_sets(self):
    value = rating_utils._cosine_similarity({1}, {2, 3})
    self.assertAlmostEqual(0, value)

  def test_cosine_similarity_equal_sets(self):
    value = rating_utils._cosine_similarity({2, 3}, {2, 3})
    self.assertAlmostEqual(1, value)

  def test_cosine_similarity_intersection(self):
    value = rating_utils._cosine_similarity({1, 2}, {2, 3})
    self.assertAlmostEqual(0.5, value)

  def test_cosine_similarity_subset(self):
    value = rating_utils._cosine_similarity({1, 2}, {1})
    self.assertAlmostEqual(1 / math.sqrt(2), value)



if __name__ == '__main__':
  unittest.main()
