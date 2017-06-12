import unittest
from unittest import mock
import os

from ps import dataset_io

class YieldRankingSetFilenamesTest(unittest.TestCase):

  def test_non_matching(self):
    non_matching_filenames = [
      'u1.base',
      'u3.test',
      'u4.validation',
      'u-Alg.out',
      'u2-.out',
      'abcakljsd',
      'u1-Alg'
    ]
    with mock.patch.object(os, 'listdir', return_value=non_matching_filenames):
      filenames_and_ids = list(dataset_io._yield_ranking_set_filenames('base/'))

    self.assertFalse(filenames_and_ids)

  def test_calls_listdir_for_right_path(self):
    with mock.patch.object(os, 'listdir', return_value=[]) as mock_listdir:
      list(dataset_io._yield_ranking_set_filenames('base/'))

    mock_listdir.assert_called_once_with('base/')

if __name__ == '__main__':
  unittest.main()
