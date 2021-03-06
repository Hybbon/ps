import collections
import concurrent.futures
import logging
import os.path

import numpy as np
import pandas as pd

from ps import dataset_io
from ps import logging_utils
from ps.metrics import eild
from ps.metrics import epc
from ps.metrics import map as map_module


def _run_metric_multi_process(metric, ranking_set_by_id, cutoff):
  with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = []
    for ranking_set in ranking_set_by_id.values():
      future = executor.submit(metric.compute, ranking_set, num_items=cutoff)
      futures.append(future)

    return [future.result() for future in futures]


def _run_metric_single_process(metric, ranking_set_by_id, cutoff):
  return [
      metric.compute(ranking_set, num_items=cutoff)
      for ranking_set in ranking_set_by_id.values()
  ]


def _compute_metric(metric_settings, ranking_set_by_id, rating_set_by_fold):
  metric = metric_settings.constructor(ranking_set_by_id, rating_set_by_fold)

  logging.info('Computing all %s', metric.NAME)

  metric_values_by_cutoff = {}
  for cutoff in metric_settings.cutoffs:
    run = (_run_metric_multi_process
           if metric_settings.multiprocess else _run_metric_single_process)
    metric_values_by_cutoff[cutoff] = run(metric, ranking_set_by_id, cutoff)

  result_records = []
  for cutoff, metric_values in metric_values_by_cutoff.items():
    for ranking_set_id, value in zip(ranking_set_by_id, metric_values):
      record = (metric.NAME, cutoff, ranking_set_id.fold, ranking_set_id.source,
                value)
      result_records.append(record)

  results = pd.DataFrame.from_records(
      result_records, columns=('metric', 'cutoff', 'fold', 'source', 'value'))

  logging.info('Done computing %s', metric.NAME)
  return results


MetricSettings = collections.namedtuple('MetricSettings', (
    'constructor', 'cutoffs', 'multiprocess'))

_METRICS = [
    MetricSettings(constructor=epc.EPC, cutoffs=[10], multiprocess=True),
    MetricSettings(constructor=eild.EILD, cutoffs=[10], multiprocess=True),
    MetricSettings(constructor=map_module.MAP, cutoffs=[10], multiprocess=True),
]


def compute_all_metrics(ranking_set_by_id, rating_set_by_fold):
  results_frames = []
  for metric_settings in _METRICS:
    frame = _compute_metric(metric_settings, ranking_set_by_id,
                            rating_set_by_fold)
    results_frames.append(frame)
  return pd.concat(results_frames)




def main(dataset_dir, output_dir):
  logging.info('Loading ranking sets')
  ranking_set_by_id = dataset_io.load_ranking_sets(dataset_dir)
  logging.info('Loading rating sets')
  rating_set_by_fold = dataset_io.load_ratings_for_all_folds(dataset_dir)
  logging_utils.log_stats_for_folds(rating_set_by_fold)
  logging.info('Done loading')
  results_frame = compute_all_metrics(ranking_set_by_id, rating_set_by_fold)
  dataset_io.save_results_frame(results_frame, output_dir)
