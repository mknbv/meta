""" Code for generation of datasets for meta-learning. """
from collections import namedtuple
from math import sqrt
import numpy as np
import numpy.linalg as nla

Sizes = namedtuple("Sizes", ["ntasks", "task_size"])
Dims = namedtuple("Dims", ["high_dim", "low_dim"])
Scales = namedtuple("Scales", ["param_scale", "label_scale"])
MetaTasks = namedtuple("MetaTasks", ["subspace", "params",
                                     "features", "labels"])


def adversarial_features(size):
  """ Samples features in an adversairal way. """
  high_dim = size if isinstance(size, int) else size[-1]
  thetas = np.linspace(0, np.pi / 2, num=high_dim)
  uniform = np.random.uniform(low=-sqrt(3), high=sqrt(3), size=size)
  normal = np.random.normal(size=size)
  return np.cos(thetas) * uniform + np.sin(thetas) * normal


def adversarial_params(size, scale=1.):
  """ Samples parameters in an adversarial way. """
  ntasks, low_dim = size
  params = np.random.normal(size=(low_dim, low_dim), scale=scale)
  indices = np.random.choice(low_dim, size=ntasks - low_dim)
  return np.concatenate([params, params[indices]], 0)


class TaskSampler:
  """ Samples dataset for meta-learning. """
  def __init__(self, subspace_dist=np.random.normal,
               params_dist=np.random.normal,
               features_dist=np.random.normal,
               label_noise_dist=np.random.normal):
    self.subspace_dist = subspace_dist
    self.params_dist = params_dist
    self.features_dist = features_dist
    self.label_noise_dist = label_noise_dist

  def subspace_params(self, ntasks, high_dim, low_dim, param_scale=1.):
    """ Samples subspace and parameter matrices. """
    orth_matrix, _, _ = nla.svd(self.subspace_dist(size=(high_dim, low_dim)))
    subspace = orth_matrix[:, :low_dim]
    params = self.params_dist(size=(ntasks, low_dim), scale=param_scale)
    return subspace, params

  def features_labels(self, subspace, params, task_size, scale=1.):
    """ Samples features and labels. """
    high_dim, ntasks = subspace.shape[0], params.shape[0]
    features = self.features_dist(size=(ntasks, task_size, high_dim))
    labels = (np.sum((features @ subspace) * params[:, None], -1)
              + self.label_noise_dist(scale=scale, size=(ntasks, task_size)))
    return features, labels

  def sample(self, sizes: Sizes, dims: Dims, scales: Scales):
    """ Samples full tasks. """
    subspace, params = self.subspace_params(sizes.ntasks, dims.high_dim,
                                            dims.low_dim, scales.param_scale)
    features, labels = self.features_labels(subspace, params, sizes.task_size,
                                            scales.label_scale)
    return MetaTasks(subspace, params, features, labels)

  def sample_with_kwargs(self, **kwargs):
    """ Binds arguments of sample to specified dict of kwargs. """
    sizes = Sizes(kwargs.pop("ntasks"), kwargs.pop("task_size"))
    dims = Dims(kwargs.pop("high_dim"), kwargs.pop("low_dim"))
    scales = Scales(kwargs.get("param_scale", 1), kwargs.get("label_scale", 1))
    if "param_scale" in kwargs:
      kwargs.pop("param_scale")
    if "label_scale" in kwargs:
      kwargs.pop("label_scale")
    if kwargs:
      raise ValueError(f"unsed kwargs={kwargs}")
    return self.sample(sizes, dims, scales)
