import io
import os
import unittest
from unittest import mock

import numpy as np

from ps import dataset_io


class YieldRankingSetFilenamesTest(unittest.TestCase):

  def test_non_matching(self):
    non_matching_filenames = [
        'u1.base', 'u3.test', 'u4.validation', 'u-Alg.out', 'u2-.out',
        'abcakljsd', 'u1-Alg'
    ]
    with mock.patch.object(os, 'listdir', return_value=non_matching_filenames):
      filenames_and_ids = list(dataset_io._yield_ranking_set_filenames('base/'))

    self.assertFalse(filenames_and_ids)

  def test_calls_listdir_for_right_path(self):
    with mock.patch.object(os, 'listdir', return_value=[]) as mock_listdir:
      list(dataset_io._yield_ranking_set_filenames('base/'))

    mock_listdir.assert_called_once_with('base/')


class LoadRatingsTest(unittest.TestCase):

  def test_loads_ratings(self):
    fake_rating_file = io.StringIO("""1\t2\t3
4\t5\t6
""")
    frame = dataset_io._load_ratings(fake_rating_file)

    self.assertTrue((frame.user_id == [1, 4]).all())
    self.assertTrue((frame.item_id == [2, 5]).all())
    self.assertTrue((frame.rating == [3, 6]).all())


class LoadRankingMatrixTest(unittest.TestCase):

  def test_load_matrix(self):
    file_contents = """1\t{2:0.63762,3:341.32871e-23}
4\t{5:4.2323,6:-3.2282}
"""

    with mock.patch(
        'ps.dataset_io.open',
        return_value=io.StringIO(file_contents)) as mock_open:
      rankings_matrix, user_ids = dataset_io._load_ranking_matrix('rankings.out')

    expected_matrix = np.array([[2, 3], [5, 6]], dtype='int32')

    self.assertTrue((expected_matrix == rankings_matrix).all())

  def test_load_different_length_rankings(self):
    file_contents = """1\t{2:0.63762}
4\t{5:4.2323,6:-3.2282}
"""

    with mock.patch(
        'ps.dataset_io.open',
        return_value=io.StringIO(file_contents)) as mock_open:
      rankings_matrix, user_ids = dataset_io._load_ranking_matrix('rankings.out')

    expected_matrix = np.array([[2, -1], [5, 6]], dtype='int32')

    self.assertTrue((expected_matrix == rankings_matrix).all())


if __name__ == '__main__':
  unittest.main()
