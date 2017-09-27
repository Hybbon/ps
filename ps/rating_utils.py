import collections
import functools
import math

import numpy as np

def _compute_by_fold(f):

  @functools.wraps(f)
  def wrapped(rating_set_by_fold, *args, **kwargs):
    results_by_fold = {
        fold: f(rating_set, *args, **kwargs)
        for fold, rating_set in rating_set_by_fold.items()
    }
    return collections.OrderedDict(sorted(results_by_fold.items()))

  return wrapped


def _compute_rated_by_user(rating_set, split_name):
  rated_by_user = collections.defaultdict(frozenset)
  split_frame = getattr(rating_set, split_name)
  for user_id, user_frame in split_frame.groupby('user_id'):
    rated_by_user[user_id] = frozenset(user_frame.item_id)
  return rated_by_user


compute_rated_by_fold = _compute_by_fold(_compute_rated_by_user)

compute_hits_by_fold = functools.partial(
    compute_rated_by_fold, split_name='test')


def _compute_items(rating_set):
  return rating_set.all_ratings().item_id.unique()


_compute_items_by_fold = _compute_by_fold(_compute_items)


def _compute_users(rating_set):
  return rating_set.all_ratings().user_id.unique()


_compute_users_by_fold = _compute_by_fold(_compute_users)



def _compute_popularity(rating_set):
  """Computes the popularity of the items in the ratings.

  ratings -- A RatingSet object with all ratings.

  Returns a Series of the popularity indexed by item_id. Popularity values
  range from 0 to 1.
  """
  num_users = len(rating_set.base.user_id.unique()) or 1  # No ratings.
  return rating_set.base.item_id.value_counts() / num_users


compute_popularity_by_fold = _compute_by_fold(_compute_popularity)


def _compute_likers(rating_set):
  return {
      item_id: frozenset(item_frame.user_id)
      for item_id, item_frame in rating_set.base.groupby(['item_id'])
  }


_compute_likers_by_fold = _compute_by_fold(_compute_likers)

def _cosine_similarity(a, b):
  return len(a & b) / ((math.sqrt(len(a)) * math.sqrt(len(b))) or 1)


def _compute_distances(k_items, l_items, likers_by_item):
  distances = []
  for k_item, l_item in zip(k_items, l_items):
    k_likers = likers_by_item.get(k_item, frozenset())
    l_likers = likers_by_item.get(l_item, frozenset())
    distance = 1 - _cosine_similarity(k_likers, l_likers)
    distances.append(distance)
  return distances


def _compute_distance_matrix(rating_set):
  likers_by_item = _compute_likers(rating_set)
  item_ids = sorted(likers_by_item.keys())
  num_items = len(item_ids)
  matrix = np.zeros((num_items, num_items))
  for i, item_i in enumerate(item_ids):
    for j, item_j in enumerate(item_ids):
      i_likers = likers_by_item.get(item_i, frozenset())
      j_likers = likers_by_item.get(item_j, frozenset())
      matrix[i][j] = 1 - _cosine_similarity(i_likers, j_likers)

  return matrix, item_ids

compute_distances_by_fold = _compute_by_fold(_compute_distance_matrix)
