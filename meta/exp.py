""" Implementation of experiment-related utilities. """
from collections import defaultdict
from copy import deepcopy
from contextlib import contextmanager, nullcontext
from functools import partial
from itertools import islice
from math import ceil, sqrt
from operator import itemgetter
import os
import pickle
from time import perf_counter
from IPython.display import display, clear_output
from ipywidgets import Output
import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np
import numpy.linalg as nla
from tqdm.auto import tqdm
from meta.alg import OracleSingleTask
from meta.plot import (update_line, refresh_axis, figure_context,
                       axes_context, plot_mean_std, semilogy_mean_std,
                       semilogx_mean_std, loglog_mean_std)
from meta.task import TaskSampler


def load(filename):
  """ Loads object from file with the specified name. """
  with open(filename, "rb") as readfile:
    return pickle.load(readfile)

def dump(obj, filename):
  """ Dumps object to the file with the specified name. """
  with open(filename, "wb") as writefile:
    pickle.dump(obj, writefile)


class Timer:
  """ Time how much running a block of code takes. """
  def __init__(self):
    self.start_time = 0.
    self.elapsed = 0.

  def __enter__(self):
    self.start_time = perf_counter()
    return self

  def __exit__(self, exc_type, exc_value, exc_tb):
    _ = exc_type, exc_value, exc_tb
    self.elapsed = perf_counter() - self.start_time


class Plotter:
  """ Base class for plotting experiments. """
  def __init__(self, output=None, fig=None):
    self.output = output
    if self.output is not None:
      display(output)
    self.fig = fig if fig is not None else plt.figure(facecolor="white")
    self.lines = {}

  @classmethod
  def jupyter(cls, *args, **kwargs):
    """ Creates plotter for plotting in Jupyter notebook. """
    return cls(*args, output=Output(), **kwargs)

  def redraw_legend(self):
    """ Redraws of the current figure. """
    for ax in self.fig.get_axes():
      legend = ax.get_legend()
      if legend is not None:
        legend.remove()
    plt.legend()

  def maybe_add_line(self, key):
    """ Adds line with the specified key if it does not exist. """
    if key not in self.lines:
      self.lines[key] = plt.plot([], label=key.split("/")[1])[0]
      self.redraw_legend()

  def before_update_line(self):
    """ Code to run before a line is updated. """
    if self.output is not None:
      clear_output(True)

  def update_line(self, line, xs, ys):
    """ Updates the given line with new x-y pairs. """
    update_line(line, xs, ys)
    for ax in self.fig.get_axes():
      refresh_axis(ax)

  def after_update_line(self):
    """ Code to run after a line is updated. """
    if self.output is not None:
      display(self.fig)

  def update(self, key, xs, ys):
    """ Updates the plot under key with xs and ys points. """
    with figure_context(self.fig):
      self.maybe_add_line(key)
      context = self.output if self.output is not None else nullcontext()
      with context:
        self.before_update_line()
        self.update_line(self.lines[key], xs, ys)
        self.after_update_line()

  def plot(self, exp, argname, dist_fn):
    """ Replots the figure of this plotter. """
    for ax in self.fig.get_axes():
      ax.clear()
    self.lines.clear()
    for val, sample, key, alg in exp.iter():
      self.update(f"{argname}/{key}", [val], [dist_fn(alg, sample)])
    self.clear_output()

  def clear_output(self):
    """ Clears output to avoid its redundant display. """
    context = self.output if self.output is not None else nullcontext()
    with context:
      clear_output()


class JointPlotter(Plotter):
  """ Class for plotting multiple plots of the same experiment. """
  def __init__(self, num, figsize=None, output=None):
    if figsize is None:
      figsize = 12, 12 / num
    fig, self.axes = plt.subplots(1, num, figsize=figsize,
                                  facecolor="white")
    super().__init__(output=output, fig=fig)

  def redraw_legend(self):
    for child in self.fig.get_children():
      if isinstance(child, mplt.legend.Legend):
        child.remove()
    self.fig.legend()

  def maybe_add_line(self, key):
    if key not in self.lines:
      self.lines[key] = [
        self.axes[i].plot([], label=key.split("/")[1] if i == 0 else None)[0]
        for i in range(len(self.axes))
      ]
      self.redraw_legend()

  def update_line(self, line, xs, ys):
    ys = [[y[i] for y in ys] for i in range(len(self.axes))]
    for i in range(len(line)):
      update_line(line[i], xs, ys[i])
    for ax in self.axes:
      refresh_axis(ax)


def frobenius_dist(alg, sample):
  """ Frobenius distance between fitted alg and meta-learning task sample. """
  target_weights = sample.params @ sample.subspace.T
  return (nla.norm(target_weights - alg.get_weights(), "fro")
          / sqrt(target_weights.shape[0]))


def subspace_dist(alg, sample):
  """ Subspace distance between fitted alg and meta-laerning task sample. """
  target = sample.subspace
  high_dim, low_dim = target.shape
  eye = np.eye(high_dim, high_dim)
  return nla.norm((eye - target @ target.T) @ alg.subspace) / sqrt(low_dim)


def sine_dist(alg, sample):
  """ Sine theta distance between fitted alg and meta-learning task sample. """
  target = sample.subspace
  estimate = alg.subspace
  singular_values = nla.svd(target.T @ estimate)[1]
  cosine = min(singular_values)
  if isinstance(alg, OracleSingleTask):
    cosine = min(1, cosine)
  return sqrt(1 - cosine ** 2)


def sines(alg, sample):
  """ Returns array of principle angles between alg and true subspace. """
  target = sample.subspace
  high_dim, low_dim = target.shape
  eye = np.eye(high_dim, high_dim)
  return np.sqrt(
    nla.svd((eye - target @ target.T) @ alg.subspace, compute_uv=False))


def frobenius_subspace_dist(alg, sample):
  """ Returns both Frobenius and subspace distances. """
  return frobenius_dist(alg, sample), subspace_dist(alg, sample)


def frobenius_sine_dist(alg, sample):
  """ Returns tuple of Frobenius and sine distances. """
  return frobenius_dist(alg, sample), sine_dist(alg, sample)


class Exp:
  """ Class corresponding to an experiment. """
  def __init__(self, algs, sampler=None, plotter=None):
    self.algs = algs
    self.sampler = sampler if sampler is not None else TaskSampler()
    self.plotter = plotter if plotter is not None else JointPlotter(2)
    self.samples = defaultdict(dict)
    self.times = defaultdict(partial(defaultdict, dict))
    self.fitted_algs = defaultdict(partial(defaultdict, dict))

  @classmethod
  def make_with_bound_sampler(cls, algs, sampler=None, plotter=None, **kwargs):
    """ Creates experiment with sampler.sample_with_kwargs binded to kwargs. """
    sampler = deepcopy(sampler) if sampler else TaskSampler()
    sampler.sample_with_kwargs = partial(sampler.sample_with_kwargs, **kwargs)
    return cls(algs, sampler, plotter)

  def run(self, argname, space, dist_fn=frobenius_subspace_dist):
    """ Runs the experiment with values of argname coming from space. """
    for val in tqdm(space):
      sample = self.sampler.sample_with_kwargs(**dict([(argname, val)]))
      self.samples[argname][val] = sample
      for name, alg in self.algs.items():
        if hasattr(alg, "set_hyperparams"):
          alg.set_hyperparams(self.sampler, sample, argname, val)
        alg.params = None
        with Timer() as timer:
          alg.fit(sample.features, sample.labels)
          if alg.params is None:
            alg.adapt(sample.features, sample.labels)
        self.times[argname][val][name] = timer.elapsed
        self.fitted_algs[argname][val][name] = deepcopy(alg)
        dist = dist_fn(alg, sample)
        self.plotter.update(f"{argname}/{name}", [val], [dist])
    self.plotter.clear_output()

  def plot(self, argname, dist_fn=frobenius_subspace_dist):
    """ Plots the results of the experiment when varying argname argument. """
    self.plotter.plot(self, argname, dist_fn)

  def iter(self, argname):
    """ Iterates over the samlpes and the fitted algs of this experiment. """
    for val, sample in sorted(self.samples[argname].items()):
      for key, alg in self.fitted_algs[argname][val].items():
        yield val, sample, key, alg

  def results(self, dist_fn=frobenius_subspace_dist):
    """ Returns dictionary of np.ndarray's of all of the results. """
    vals = defaultdict(list)
    results = defaultdict(list)
    for argname in self.samples:
      vals[argname] = np.asarray(sorted(self.samples[argname].keys()))
      for val, sample, key, alg in self.iter(argname):
        results[f"{argname}/{key}"].append(dist_fn(alg, sample))
    vals, results = dict(vals), dict(results)
    for key, val in results.items():
      results[key] = np.asarray(val)
    return vals, results


class BatchedExp:
  """ Batch of experiments. """
  def __init__(self, exps):
    if not exps:
      raise ValueError("exps cannot be empty")
    self.exps = exps

  @classmethod
  def from_dir(cls, dirname, filename="exp.pickle",
               batch_size=12, ignore_not_found=False):
    """ Loads experiments from the directory with the specified name. """
    filenames = set(os.listdir(dirname))
    exps = []
    for i in range(batch_size):
      new_fname = f"{i:02d}-{filename}"
      if new_fname not in filenames:
        if ignore_not_found:
          continue
        raise ValueError(f"cannot find file with name '{new_fname}' "
                         f"in directory {dirname}")
      exps.append(load(os.path.join(dirname, new_fname)))
      plt.close()
    return cls(exps)

  def results(self, dist_fn=frobenius_subspace_dist):
    """ Plots the results of this batched experiment. """
    exp = self.exps[0]
    vals, results = exp.results(dist_fn)
    for key, val in results.items():
      results[key] = [val]
    for exp in islice(self.exps, 1, None):
      exp_vals, exp_results = exp.results(dist_fn)
      for key in vals:
        if key in exp_vals and np.any(vals[key] != exp_vals[key]):
          raise ValueError(f"exps have mismatching vals under {key}: "
                           f"{vals[key]} != {exp_vals[key]}")
        exp_vals.pop(key)
      vals.update(exp_vals)
      for key in exp_results:
        if key in results and len(results[key][0]) != len(exp_results[key]):
          raise ValueError(f"exp results have mismatching lengths under {key}: "
                           "{len(results[key][0])} != {len(exp_results[key])})")
        results[key].append(exp_results[key])

    for key, val in results.items():
      results[key] = np.asarray(val)
    return vals, results

  def times(self, argname):
    """ Returns dict with keys=algnames and values=times of alg. """
    vals = sorted(self.exps[0].samples[argname].keys())
    times = defaultdict(list)
    for i, exp in enumerate(self.exps):
      if vals != (new_vals := sorted(exp.samples[argname].keys())):
        raise ValueError(f"values of exp[0]={vals} mismatch with"
                         f"values of exp[{i}]={new_vals}")
      new_times = defaultdict(list)
      for _, algs_times in sorted(exp.times["ntasks"].items(),
                                  key=itemgetter(0)):
        for key in algs_times:
          new_times[key].append(algs_times[key])
      for key in new_times:
        times[key].append(new_times[key])
    for key in times:
      times[key] = np.asarray(times[key])
    return vals, times

  def plot_results(self, argname, dist_fn,
                   plot_fn=semilogy_mean_std,
                   disable_algs=None):
    """ Plots batched experiment results."""
    if not disable_algs:
      disable_algs = set()
    disable_algs = set(disable_algs)
    vals, results = self.results(dist_fn)
    for key in self.exps[0].algs:
      if key not in disable_algs:
        plot_fn(vals[argname], results[f"{argname}/{key}"], label=key)
    plt.legend()


class FinalPlotter:
  """ Class to perform final plotting of the results. """
  def __init__(self, keys_to_labels=None, colors=None,
               disable_algs=(("single-lasso",),
                             ("single-lasso", "single-oracle")),
               markers=None):
    if keys_to_labels is None:
      keys_to_labels = {"single-oracle": "oracle", "nuc-val": "nuc"}
    self.keys_to_labels = keys_to_labels
    if colors is None:
      _, colors = FinalPlotter.get_default_colors()
    self.colors = colors
    self.disable_frobenius = set(disable_algs[0])
    self.disable_sine = set(disable_algs[1])
    if markers is None:
      markers = {"bm": 4, "altmin": 5}
    self.markers = markers
    self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="white")
    plt.close()

  @staticmethod
  def get_default_colors():
    """ Returns the colors of the default color scheme and dict of colors. """
    color_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {
      "mom": color_list[0],
      "nuc": color_list[1],
      "nuc-val": color_list[1],
      "single": color_list[2],
      "oracle": color_list[3],
      "altmin": color_list[4],
      "bm": color_list[5],
    }
    return color_list, colors

  @staticmethod
  def plot_mean_std_fn(**kwargs):
    """ Default plot_mean_std plotting function. """
    return partial(plot_mean_std, linewidth=1., alpha=(1., 0.1), **kwargs)

  @staticmethod
  def semilogx_mean_std_fn(**kwargs):
    """ Default plotting function with log X-axis. """
    return partial(semilogx_mean_std, linewidth=1., alpha=(1., 0.1), **kwargs)

  @staticmethod
  def semilogy_mean_std_fn(**kwargs):
    """ Default plotting function with log Y-axis. """
    return partial(semilogy_mean_std, linewidth=1., alpha=(1., 0.1), **kwargs)

  @staticmethod
  def loglog_mean_std_fn(**kwargs):
    """ Default plotting function with log X- and Y-axes. """
    return partial(loglog_mean_std, linewidth=1., alpha=(1., 0.1), **kwargs)

  def clear(self, xlabel=None):
    """ Clears the axes of this plotter. """
    for ax in self.axes:
      ax.clear()
      ax.set_xlabel(xlabel)
      ax.grid()
    self.axes[0].set_ylabel("Normalized Frobenius distance")
    self.axes[1].set_ylabel("Sine distance")
    for child in self.fig.get_children():
      if isinstance(child, mplt.legend.Legend):
        child.remove()

  @contextmanager
  def context(self, ax_index=None):
    """ Context manager for underlying figures and axes. """
    with figure_context(self.fig):
      if ax_index is not None:
        with axes_context(self.axes[ax_index]):
          yield
      else:
        yield

  def legend(self):
    """ Adds figure legend. """
    for child in self.fig.get_children():
      if isinstance(child, mplt.legend.Legend):
        child.remove()
    handles, labels = self.axes[0].get_legend_handles_labels()
    new_handles, new_labels = self.axes[1].get_legend_handles_labels()
    handles.extend(new_handles)
    labels.extend(new_labels)
    argsort = np.argsort(labels)
    handles, labels = np.asarray(handles)[argsort], np.asarray(labels)[argsort]
    self.fig.legend(handles, labels)

  def set_xticks(self, ax_index, vals, max_vals=10):
    """ Sets xticks to values. """
    if len(vals) <= max_vals:
      self.axes[ax_index].set_xticks(vals)
    else:
      self.axes[ax_index].set_xticks(vals[::ceil(len(vals) / max_vals)])
    self.axes[ax_index].get_xaxis().set_major_formatter(
      mplt.ticker.ScalarFormatter())

  def plot(self, exp, argname, plot_fn=None, disable_algs=None):
    """ Plots the distance function of the given experment. """
    if plot_fn is None:
      plot_fn = FinalPlotter.semilogx_mean_std_fn()
    if disable_algs is None:
      disable_algs = ((), ())
    disable_frobenius = self.disable_frobenius | set(disable_algs[0])
    disable_sine = self.disable_sine | set(disable_algs[1])
    frobenius_plots, sine_plots = [], []
    vals, results = exp.results(frobenius_sine_dist)
    argname, xlabel = ((argname, argname) if isinstance(argname, str)
                       else argname)
    self.clear(xlabel)
    for key in sorted(exp.exps[0].algs):
      if key in disable_frobenius & disable_sine:
        continue
      label = self.keys_to_labels.get(key, key)
      color = self.colors[label]
      if key not in disable_frobenius:
        with self.context(0):
          frobenius_plots.append(
            plot_fn(vals[argname], results[f"{argname}/{key}"][..., 0],
                    marker=self.markers.get(label, 'o'), label=label,
                    ylim=(0, np.inf), color=color)
          )
          if self.axes[0].get_xscale() == "log":
            self.set_xticks(0, vals[argname])
          label = None
      if key not in disable_sine:
        with self.context(1):
          sine_plots.append(
            plot_fn(vals[argname], results[f"{argname}/{key}"][..., 1],
                    marker=self.markers.get(label, 'o'), label=label,
                    ylim=(0, 1), color=color)
          )
          if self.axes[1].get_xscale() == "log":
            self.set_xticks(1, vals[argname])
    self.legend()
    plt.close()
    return frobenius_plots, sine_plots

  def plot_range(self, exp, key, indices,
                 plot_fn=partial(plot_mean_std, linewidth=1., alpha=(1., 0.1))):
    """ Plots subset of the values for frobenius distance. """
    argname, key = key.split("/")
    label = self.keys_to_labels.get(key, key)
    color = self.colors[label]
    with self.context(0):
      vals, results = exp.results(frobenius_dist)
      plot_fn(vals[argname][indices], results[f"{argname}/{key}"][:, indices],
              marker=self.markers.get(label, 'o'), label=label,
              ylim=(0, np.inf), color=color)
    plt.close()

  def savefig(self, filename, format="pdf", dpi=1200, bbox_inches=0):
    """ Saves the figure to a file. """
    self.fig.savefig(filename, format=format, dpi=dpi, bbox_inches=bbox_inches)
