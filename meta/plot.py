""" Plot utilities. """
from contextlib import contextmanager
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mstats


def update_line(line, newxs, newys):
  """ Updates line with new points. """
  xs, ys = map(list, line.get_data())
  xs.extend(newxs)
  ys.extend(newys)
  line.set_data(xs, ys)
  return line


def refresh_axis(ax=None):
  """ Refreshes axis. """
  if ax is None:
    ax = plt.gca()
  ax.relim()
  ax.autoscale_view()


@contextmanager
def figure_context(figure):
  """ Sets give figure as default. """
  default_fig = plt.gcf().number
  try:
    yield plt.figure(figure.number)
  finally:
    plt.figure(default_fig)


@contextmanager
def axes_context(axes):
  """ Sets given axes as default. """
  default_axes = plt.gca()
  try:
    yield plt.sca(axes)
  finally:
    plt.sca(default_axes)


def set_figure_settings(xlabel, ylabel, grid=True, title=None, legend=True):
  """ Setups up current figure. """
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.grid(grid)
  if title is not None:
    plt.title(title)
  if legend:
    plt.legend()


# pylint: disable=invalid-name,too-many-arguments
def _plot_median_quantiles(xplot_fn, ys, probs=(0.25, 0.75), axis=0,
                           label=None, alpha=0.3):
  """ Plots median and quantiles. """
  if len(probs) != 2:
    raise ValueError(f"probs must have length 2, got len(probs)={len(probs)}")
  probs = [probs[0], 0.5, probs[1]]
  quantiles = mstats.mquantiles(ys, probs, axis=axis)
  line, = xplot_fn(quantiles[1], label=label)
  xs, = xplot_fn.args
  fill = plt.fill_between(xs, quantiles[0], quantiles[2],
                          color=line.get_color(), alpha=alpha)
  return line, fill


def plot_median_quantiles(x, ys, probs=(0.25, 0.75),
                        axis=0, label=None, alpha=0.3):
  """ Plot quantiles. """
  _plot_median_quantiles(partial(plt.plot, x),
                         ys, probs, axis, label, alpha)

def semilogy_median_quantiles(x, ys, probs=(0.25, 0.75),
                            axis=0, label=None, alpha=0.3):
  """ Y-axis log plot of quantiles. """
  _plot_median_quantiles(partial(plt.semilogy, x),
                         ys, probs, axis, label, alpha)

def semilogx_median_quantiles(x, ys, probs=(0.25, 0.75),
                            axis=0, label=None, alpha=0.3):
  """ X-axis log plot of quantiles. """
  _plot_median_quantiles(partial(plt.semilogx, x),
                         ys, probs, axis, label, alpha)

def loglog_median_quantiles(x, ys, probs=(0.25, 0.75),
                          axis=0, label=None, alpha=0.3):
  """ Log-log plot of quantiles. """
  _plot_median_quantiles(partial(plt.loglog, x),
                         ys, probs, axis, label, alpha)


def _plot_mean_std(xplot_fn, ys, axis=0, fill=True,
                   ylim=(-np.inf, np.inf), label=None,
                   alpha=(1, 0.3), **kwargs):
  """ Plots mean and standard deviation area. """
  mean = np.mean(ys, axis)
  std = np.std(ys, axis)
  if isinstance(alpha, (float, int)):
    alpha = (alpha, alpha)
  line, = xplot_fn(mean, label=label, alpha=alpha[0], **kwargs)
  x, = xplot_fn.args
  kwargs.pop("color", None)
  kwargs.pop("marker", None)
  if fill:
    fill_result = plt.fill_between(x,
                                   np.maximum(ylim[0], mean - std),
                                   np.minimum(ylim[1], mean + std),
                                   color=line.get_color(),
                                   alpha=alpha[1], **kwargs)
    return line, fill_result
  low, high = plt.plot(x,
                       np.maximum(ylim[0], mean - std),
                       np.minimum(ylim[1], mean + std),
                       color=line.get_color(),
                       alpha=alpha[1], **kwargs)
  return mean, low, high



def plot_mean_std(x, ys, axis=0, fill=True, ylim=(-np.inf, np.inf),
                  label=None, alpha=(1., 0.3), **kwargs):
  """ Plots mean and standard deviation area. """
  return _plot_mean_std(partial(plt.plot, x), ys, axis, fill=fill,
                        ylim=ylim, label=label, alpha=alpha, **kwargs)

def semilogy_mean_std(x, ys, axis=0, fill=True, ylim=(-np.inf, np.inf),
                      label=None, alpha=(1., 0.3), **kwargs):
  """ Y-axis log plot of mean and standard deviation area. """
  return _plot_mean_std(partial(plt.semilogy, x), ys, axis, fill=fill,
                        ylim=ylim, label=label, alpha=alpha, **kwargs)

def semilogx_mean_std(x, ys, axis=0, fill=True, ylim=(-np.inf, np.inf),
                      label=None, alpha=(1., 0.3), **kwargs):
  """ X-axis log plot of mean and standard deviation area. """
  return _plot_mean_std(partial(plt.semilogx, x), ys, axis, fill=fill,
                        ylim=ylim, label=label, alpha=alpha, **kwargs)

def loglog_mean_std(x, ys, axis=0, fill=True, ylim=(-np.inf, np.inf),
                    label=None, alpha=(1., 0.3), **kwargs):
  """ Log-log plot of mean and standard deviation area. """
  return _plot_mean_std(partial(plt.loglog, x), ys, axis, fill=fill,
                        ylim=ylim, label=label, alpha=alpha, **kwargs)


def _plot_mean_lines(plot_fn, ys, axis=0, label=None, alpha=0.3):
  """ Plots all lines and their mean using plot_fn. """
  ys = np.asarray(ys)
  line, = plot_fn(np.mean(ys, axis), label=label, lw=3)
  n = ys.shape[axis]
  for i in range(n):
    plot_fn(np.take(ys, i, axis), color=line.get_color(), alpha=alpha)

def plot_mean_lines(x, ys, axis=0, label=None, alpha=0.3):
  """ Plots all lines and their mean. """
  _plot_mean_lines(partial(plt.plot, x), ys, axis, label, alpha)

def semilogy_mean_lines(x, ys, axis=0, label=None, alpha=0.3):
  """ Y-axis log plot of lines and their mean. """
  _plot_mean_lines(partial(plt.semilogy, x), ys, axis, label, alpha)

def semilogx_mean_lines(x, ys, axis=0, label=None, alpha=0.3):
  """ X-axis log plot of lines and their mean. """
  _plot_mean_lines(partial(plt.semilogx, x), ys, axis, label, alpha)

def loglog_mean_lines(x, ys, axis=0, label=None, alpha=0.3):
  """ Log-log plot of lines and their mean. """
  _plot_mean_lines(partial(plt.loglog, x), ys, axis, label, alpha)


def mean_std_errorbar(x, ys, axis=0, **kwargs):
  """ Plots mean and standard deviation error bars around it. """
  mean, std = np.mean(ys, axis), np.std(ys, axis)
  return plt.errorbar(x, mean, yerr=std, **kwargs)

# pylint: enable=invalid-name, too-many-arguments
