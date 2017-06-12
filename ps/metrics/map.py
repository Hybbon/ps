import collections


def _compute_hits_by_user(rating_set):
  hits_by_user = {}
  for user_id, user_frame in rating_set.test.groupby('user_id'):
    hits_by_user[user_id] = set(user_frame.item_id)
  return hits_by_user


def _compute_hits_by_fold(rating_set_by_fold):
  hits_by_fold = {
      fold: _compute_hits_by_user(rating_set)
      for fold, rating_set in rating_set_by_fold.items()
  }
  return collections.OrderedDict(sorted(hits_by_fold.items()))


def _precision(ranking, hits):
  if not hits:
    return 0

  if len(ranking) == 0:  # Can't check for truthiness -- numpy array.
    raise ValueError('Empty ranking.')

  num_hits = 0
  total = 0

  for i, item_id in enumerate(ranking):
    if item_id in hits:
      num_hits += 1
      total += num_hits / (i + 1)

  return total / min(len(ranking), len(hits))


class MAP(object):
  NAME = 'MAP'

  def __init__(self, ranking_set_by_id, rating_set_by_fold):
    self.hits_by_fold = _compute_hits_by_fold(rating_set_by_fold)

  def compute(self, ranking_set, num_items=None):
    hits_by_user = self.hits_by_fold[ranking_set.id.fold]

    matrix = ranking_set.matrix

    if num_items is not None:
      matrix = matrix[:, :num_items]
    else:
      num_items = matrix.shape[1]

    total = 0
    for user_ranking, user_id in zip(matrix, ranking_set.user_ids):
      user_hits = hits_by_user.get(user_id)
      if user_hits is None:
        user_hits = set()
      total += _precision(user_ranking, user_hits)

    return total / len(ranking_set.user_ids)
