import collections

class Oracle(object):

  def __init__(self, ranking_set_by_id, rating_set_by_fold):
    super().__init__()
    self.ranking_set_by_id = ranking_set_by_id
    self.rating_set_by_fold = rating_set_by_fold

  def _yield_ranking_set_in_fold(self, fold):
    for ranking_set in self.ranking_set_by_id.values():
      if ranking_set.id.fold == fold:
        yield ranking_set

  def _compute_recommended_in_fold(self, fold, cutoff):
    items_recommended_to_user = collections.defaultdict(set)

    for ranking_set in self._yield_ranking_set_in_fold(fold):

      matrix = ranking_set.matrix[:, :cutoff]

      for user_ranking, user_id in zip(matrix, ranking_set.user_ids):
        items_recommended_to_user[user_id].update(user_ranking)

    return collections.OrderedDict(sorted(items_recommended_to_user.items()))
