import collections
import logging
import os.path

from ps import dataset_io
from ps import gen_metrics
from ps import logging_utils

from ps.oracles import eild_oracle
from ps.oracles import epc_oracle
from ps.oracles import map_oracle

_ORACLES = [
  eild_oracle.EILDOracle,
  epc_oracle.EPCOracle,
  map_oracle.MAPOracle,
]


def _compute_oracle_rankings(oracle_constructor, ranking_set_by_id,
                             rating_set_by_fold):
  logging.info('Creating oracle rankings for %s', str(oracle_constructor))

  oracle = oracle_constructor(ranking_set_by_id, rating_set_by_fold)

  folds = rating_set_by_fold.keys()

  return [oracle.compute_optimal_ranking_set(fold, input_cutoff=20, output_cutoff=10)
          for fold in folds]



def _compute_all_oracle_rankings(ranking_set_by_id, rating_set_by_fold):
  logging.info('Creating oracle rankings for all metrics')

  created_ranking_sets_by_id = {}
  for oracle_constructor in _ORACLES:
    created_ranking_sets = _compute_oracle_rankings(oracle_constructor, ranking_set_by_id,
                            rating_set_by_fold)
    for ranking_set in created_ranking_sets:
      created_ranking_sets_by_id[ranking_set.id] = ranking_set
  return created_ranking_sets_by_id


def _build_row_string(ranking, user_id):
  item_ids_and_scores = zip(ranking, range(len(ranking), 0, -1))
  position_strings = (f'{int(item_id)}:{float(score)}'
                      for item_id, score in item_ids_and_scores)
  ranking_inside_brackets = '[' + ','.join(position_strings) + ']'

  return f'{user_id}\t{ranking_inside_brackets}'


def _save_ranking_set_to_file(ranking_set, f):
  for ranking, user_id in zip(ranking_set.matrix, ranking_set.user_ids):
    row_string = _build_row_string(ranking, user_id)
    print(row_string, file=f)


def _save_ranking_sets(ranking_set_by_id, output_dir):
  for ranking_set_id, ranking_set in ranking_set_by_id.items():
    output_path = os.path.join(
        output_dir, f'u{ranking_set_id.fold}-{ranking_set_id.source}.out')
    with open(output_path, 'w') as f:
      _save_ranking_set_to_file(ranking_set, f)


def main(dataset_dir, output_dir):
  logging.info('Loading ranking sets')
  ranking_set_by_id = dataset_io.load_ranking_sets(dataset_dir)
  logging.info('Loading rating sets')
  rating_set_by_fold = dataset_io.load_ratings_for_all_folds(dataset_dir)
  logging_utils.log_stats_for_folds(rating_set_by_fold)
  logging.info('Done loading')
  created_ranking_sets_by_id = _compute_all_oracle_rankings(ranking_set_by_id, rating_set_by_fold)
  _save_ranking_sets(created_ranking_sets_by_id, output_dir)
  results_frame = gen_metrics.compute_all_metrics(created_ranking_sets_by_id, rating_set_by_fold)
  print(results_frame)

