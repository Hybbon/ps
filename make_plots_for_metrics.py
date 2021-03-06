import argparse
import collections
import datetime
import functools
import logging
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

from ps import logging_utils

plt.rcParams['figure.figsize'] = (10, 6)


def _read_metrics_from_file(metrics_csv_path):
  return pd.DataFrame.from_csv(metrics_csv_path)


def _mean_value_by_metric(metrics, metric_name):
  eild_values = metrics[metrics.metric == metric_name][[
      'fold', 'source', 'value'
  ]]
  return eild_values.groupby('source').mean().value


def _compute_mean_metrics(metrics):
  sources_by_category = {
      'recommender': [
          'BPRSLIM', 'CofiRank', 'Hybrid_librec', 'LDA_librec', 'libfm',
          'MultiCoreBPRMF', 'RankALS_librec', 'WRMF', 'CoFactor', 'FISM_librec',
          'ItemKNN', 'LeastSquareSLIM', 'MostPopular', 'Poisson',
          'SoftMarginRankingMF'
      ],
      'era_multiobj': ['ERALinear30', 'ERALinear40', 'ERALinear50'],
      'unsup_agg': ['BordaCount', 'MedianRankAggregation'],
      'oracle': ['EPCOracle', 'MAPOracle', 'EILDOracle'],
      'era_singleobj': ['GPRA_best'],
  }
  source_category = {}
  for category, sources in sources_by_category.items():
    for source in sources:
      source_category[source] = category

  mean_metrics = pd.DataFrame({
      'map': _mean_value_by_metric(metrics, 'MAP'),
      'eild': _mean_value_by_metric(metrics, 'EILD'),
      'epc': _mean_value_by_metric(metrics, 'EPC')
  })
  mean_metrics['source_category'] = [
      source_category.get(source, 'sup_agg') for source in mean_metrics.index
  ]

  sort_weights = {
      'recommender': 0,
      'unsup_agg': 1,
      'sup_agg': 2,
      'era_singleobj': 2,
      'era_multiobj': 3,
      'oracle': 4,
  }
  mean_metrics['sort_weights'] = [
      sort_weights[category] for category in mean_metrics.source_category
  ]
  mean_metrics = mean_metrics.sort_values(by='sort_weights')
  del mean_metrics['sort_weights']
  return mean_metrics


def plot(f):

  @functools.wraps(f)
  def wrapped(path, *args, **kwargs):
    plt.clf()
    mpl.rc('font', size=15.)

    f(*args, **kwargs)

    plt.tight_layout()
    plt.savefig(path)
    plt.clf()

  return wrapped


def get_metric_display_name(internal_name):
  display_name_by_internal = {
      'map': 'Precisão',
      'eild': 'Diversidade',
      'epc': 'Novidade'
  }

  return display_name_by_internal.get(internal_name, internal_name)


@plot
def _bar_plot_for_mean_metrics(mean_metrics):
  ax = mean_metrics.plot(kind='bar')
  handles, labels = ax.get_legend_handles_labels()
  display_labels = [get_metric_display_name(label) for label in labels]
  ax.legend(handles, display_labels)


CategorySettings = collections.namedtuple('CategorySettings',
                                          ['color', 'marker', 'label'])

_CATEGORY_SETTINGS = {
    'recommender':
    CategorySettings(color='green', marker='s', label='Recommender'),
    'sup_agg':
    CategorySettings(color='darkorchid', marker='d', label='Supervised Agg.'),
    'unsup_agg':
    CategorySettings(color='darkcyan', marker='v', label='Unsup. Agg.'),
    'era_multiobj':
    CategorySettings(color='darkblue', marker='P', label='Linear ERA'),
    'era_singleobj':
    CategorySettings(color='black', marker='o', label='Precision ERA'),
    'oracle':
    CategorySettings(color='orangered', marker='x', label='Oracle'),
}


@plot
def _scatter_plot_by_source_category(mean_metrics,
                                     x,
                                     y,
                                     front_metrics=None,
                                     plot_median_x=False,
                                     plot_median_y=False,
                                     fig=None):
  if fig is None:
    fig = plt.figure()
  ax = fig.add_subplot(111)

  if front_metrics is not None:
    front_metrics.plot.scatter(
        x, y, s=45., c='red', marker='*', label='SPEA Front', ax=ax)

  if plot_median_x:
    mean_x_value = mean_metrics[x].median()
    plt.axvline(x=mean_x_value, c=(0, 0, 0, 0.65))

  if plot_median_y:
    mean_y_value = mean_metrics[y].median()
    plt.axhline(y=mean_y_value, c=(0, 0, 0, 0.65))

  for category, frame in mean_metrics.groupby('source_category'):
    settings = _CATEGORY_SETTINGS[category]
    frame.plot.scatter(
        x,
        y,
        s=45.,
        c=settings.color,
        marker=settings.marker,
        label=settings.label,
        ax=ax)

  ax.set_xlabel(get_metric_display_name(x))
  ax.set_ylabel(get_metric_display_name(y))


@plot
def _3d_scatter_plot(mean_metrics, front_metrics=None, fig=None):
  if fig is None:
    fig = plt.figure()

  threedee = fig.gca(projection='3d')
  # threedee.scatter(mean_metrics.map, mean_metrics.epc, mean_metrics.eild)

  colors = ['darkblue', 'orangered', 'green', 'darkorchid', 'darkcyan']
  markers = ['o', 'x', 's', 'd', 'x']
  for (category, frame), color, marker in zip(
      mean_metrics.groupby('source_category'), colors, markers):
    if category == 'oracle':
      continue
    threedee.scatter(
        frame.map,
        frame.epc,
        frame.eild,
        s=45.,
        c=color,
        marker=marker,
        label=category)
  # scatter_plot = frame.plot.scatter(
  #     x, y, s=45., c=color, marker=marker, label=category, ax=ax)

  if front_metrics is not None:
    threedee.scatter(
        front_metrics.map,
        front_metrics.epc,
        front_metrics.eild,
        s=45.,
        c='red',
        marker='o',
        label='Front')

  threedee.set_xlabel('map')
  threedee.set_ylabel('nov')
  threedee.set_zlabel('div')


ScatterPlotSettings = collections.namedtuple('ScatterPlotSettings', (
    'x_metric', 'y_metric', 'plot_median_x', 'plot_median_y'))


def _make_plots(mean_metrics, output_dir, front_metrics=None, extension='pdf'):
  bar_plot_path = os.path.join(output_dir, f'mean_metrics_bar_plot.{extension}')
  logging.info('Plotting mean metrics into a bar plot at %s', bar_plot_path)
  _bar_plot_for_mean_metrics(bar_plot_path, mean_metrics)

  scatter_plots = [
      ScatterPlotSettings(
          'map', 'epc', plot_median_x=False, plot_median_y=True),
      ScatterPlotSettings(
          'map', 'eild', plot_median_x=False, plot_median_y=True),
      ScatterPlotSettings(
          'epc', 'eild', plot_median_x=True, plot_median_y=True),
  ]

  for x_metric, y_metric, plot_median_x, plot_median_y in scatter_plots:
    scatter_plot_path = os.path.join(
        output_dir, f'{y_metric}_by_{x_metric}_scatter_plot.{extension}')
    logging.info('Plotting %s by %s into a scatter plot to %s', y_metric,
                 x_metric, scatter_plot_path)
    _scatter_plot_by_source_category(scatter_plot_path, mean_metrics, x_metric,
                                     y_metric, front_metrics, plot_median_x,
                                     plot_median_y)

  threedee_scatter_path = os.path.join(output_dir, f'3d_scatter.{extension}')
  logging.info('Plotting 3d scatter plot to %s', threedee_scatter_path)
  _3d_scatter_plot(threedee_scatter_path, mean_metrics, front_metrics)


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('metrics_csv_path')
  p.add_argument('--front_csv_path')
  p.add_argument('--fold', type=int)
  p.add_argument('--no_oracles', action='store_true')
  p.add_argument('--min_precision', type=float)
  p.add_argument('--keep_within_x_stds', type=float)
  return p.parse_args()


def _setup_loggers():
  time_string = datetime.datetime.now().strftime('%Y-%m-%d-%Hh%Mm%Ss')
  run_string = 'make_plots_for_metrics-{time}'.format(time=time_string)
  log_path = os.path.join('log', run_string, 'run.log')
  log_dir = os.path.dirname(log_path)

  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  logging_utils.setup_loggers(log_path)

  logging.info('Logging to %s', log_path)


def main():
  _setup_loggers()
  args = parse_args()

  output_dir = os.path.dirname(args.metrics_csv_path)

  metrics = _read_metrics_from_file(args.metrics_csv_path)
  logging.info('Done reading metrics from CSV')

  if args.front_csv_path is None:
    logging.info('No front metrics path provided')
    front_metrics = None
  else:
    logging.info('Reading front metrics from %s', args.front_csv_path)
    front_metrics = _read_metrics_from_file(args.front_csv_path)

  if args.fold is not None:
    logging.info('Filtering metrics to fold %d', args.fold)
    metrics = metrics[metrics.fold == args.fold]

  logging.info('Using metrics at %s', args.metrics_csv_path)

  logging.info('Computing mean metrics')
  mean_metrics = _compute_mean_metrics(metrics)
  logging.info('Done computing mean metrics')

  mean_metrics_path = os.path.join(output_dir, 'mean_metrics.csv')
  logging.info('Saving mean metrics to %s', mean_metrics_path)
  mean_metrics.to_csv(mean_metrics_path)

  if args.no_oracles:
    logging.info('Removing oracles before plotting')
    mean_metrics = mean_metrics[mean_metrics.source_category != 'oracle']

  if args.keep_within_x_stds is not None:
    max_precision = mean_metrics.map.max()
    std = mean_metrics.map.std()
    min_precision = max_precision - args.keep_within_x_stds * std
  elif args.min_precision is not None:
    min_precision = args.min_precision
  else:
    min_precision = None

  if min_precision is not None:
    logging.info('Filtering points with MAP under %f', min_precision)
    mean_metrics = mean_metrics[mean_metrics.map >= min_precision]
    if front_metrics is not None:
      front_metrics = front_metrics[front_metrics.map >= min_precision]

  logging.info('Plotting stuff')
  _make_plots(mean_metrics, output_dir, front_metrics)


if __name__ == '__main__':
  main()
