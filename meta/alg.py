""" Implementation of meta-learning algorithms. """
from abc import ABC, abstractmethod
from functools import partial
from math import sqrt
import jax
from jax.config import config
import jax.numpy as jnp
import numpy as np
import numpy.linalg as nla
from scipy import optimize
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV, train_test_split
config.update("jax_enable_x64", True)


class MetaLearningAlg(ABC):
  """ Base class for meta-learning algorithms. """
  def __init__(self):
    self.subspace = None
    self.params = None

  def get_weights(self):
    """ Returns estimate of the d by T weights matrix. """
    if hasattr(self, "weights"):
      return self.weights
    if self.subspace is None:
      raise ValueError("self.subsapce is None, call fit to estimate "
                       "subspace before callling this method")
    if self.params is None:
      raise ValueError("self.params is None, call fit to estimate "
                       "params before calling this method")
    return self.params @ self.subspace.T

  @abstractmethod
  def fit(self, features, labels):
    """ Performs subspace estimation on the specified data. """

  def adapt(self, features, labels):
    """ Adapts to the specified data. """
    if self.subspace is None:
      raise ValueError("self.subspace is None, call fit to estimate "
                       "subspace before calling this method")
    self.params = np.asarray([nla.lstsq(X @ self.subspace, y, rcond=None)[0]
                              for X, y in zip(features, labels)])
    return self.params



def top_eigvecs(matrix, num):
  """ Returns the matrix of num top eigenvectors. """
  eigvals, eigvecs = nla.eig(matrix)
  argsort = np.argsort(-eigvals)
  return eigvecs[:, argsort[:num]]


class MoM(MetaLearningAlg):
  """ Method of moments meta-learning algorithm. """
  def __init__(self, low_dim):
    super().__init__()
    self.low_dim = low_dim

  def fit(self, features, labels):
    ntasks, high_dim = features.shape[0], features.shape[-1]
    estimate = np.zeros((high_dim, high_dim))
    num_samples = 0
    for i in range(ntasks):
      features_labels = features[i].T * labels[i]
      estimate += features_labels @ features_labels.T
      num_samples += features[i].shape[0]
    estimate /= num_samples
    self.subspace = top_eigvecs(estimate, self.low_dim)


def mean_squared_error(features, labels, subspace, params):
  """ Computes squared error. """
  return np.mean(np.square(
    labels - np.sum(features @ subspace * params[:, None], -1)
  ))


class AltMin(MetaLearningAlg):
  """ Alternating minimization meta-learning algorithm. """
  def __init__(self, low_dim, tol=1e-6, maxiter=None):
    super().__init__()
    self.low_dim = low_dim
    self.tol = tol
    self.maxiter = maxiter if maxiter is not None else float("inf")
    self.niter = 0

  def fit(self, features, labels):
    _, _, high_dim = features.shape
    self.subspace = nla.qr(np.random.normal(size=(high_dim, self.low_dim)))[0]
    self.niter, error, delta = 0, float("inf"), float("inf")
    while self.niter < self.maxiter and delta >= self.tol:
      self.params = np.asarray([
        nla.lstsq(X, y, rcond=None)[0]
        for X, y in zip(features @ self.subspace, labels)
      ])
      params_features = np.reshape(
        self.params[:, None, :, None] @ features[:, :, None],
        (np.prod(features.shape[:2]), -1)
      )
      subspace = nla.lstsq(params_features, np.ravel(labels), rcond=None)[0]
      self.subspace = nla.qr(
        np.reshape(subspace, (self.low_dim, high_dim)).T)[0]
      new_error = mean_squared_error(features, labels,
                                     self.subspace, self.params)
      delta = abs(new_error - error)
      error = new_error
      self.niter += 1

    self.params = np.asarray([
      nla.lstsq(X, y, rcond=None)[0]
      for X, y in zip(features @ self.subspace, labels)
    ])


def nuclear_norm_loss(weights, features, labels, reg_coef=1.):
  """ Comptues (differentiable) nuclear norm regularized loss. """
  ntasks, task_size, high_dim = features.shape
  weights = jnp.reshape(weights, (ntasks, high_dim))
  prediction_error = jnp.square(
    jnp.linalg.norm(labels - jnp.sum(features * weights[:, None], -1))
  ) / (ntasks * task_size)
  regularization = jnp.linalg.norm(weights, ord="nuc") /  sqrt(ntasks)
  return prediction_error + reg_coef * regularization


class NucNorm(MetaLearningAlg):
  """ Nuclear norm regularized least squares algorithm. """
  def __init__(self, low_dim, reg_coef=None):
    super().__init__()
    self.low_dim = low_dim
    self.reg_coef = reg_coef
    self.weights = None
    self.optim_res = None

  def get_params(self, deep=True):
    """ Returns dictionary with hyperparameters of this estimator. """
    _ = deep
    return dict(reg_coef=self.reg_coef,
                low_dim=self.low_dim)

  def set_params(self, **params):
    """ Sets hyperparameters of this estimator. """
    if "reg_coef" in params:
      self.reg_coef = params.pop("reg_coef")
    if "low_dim" in params:
      self.low_dim = params.pop("low_dim")
    if params:
      raise ValueError(f"unknown params={params}")
    return self

  def set_hyperparams(self, sampler, sample, argname, val):
    """ Sets regularization coefficient based on sigma, features, labels. """
    ntasks, task_size, high_dim = sample.features.shape
    label_scale = val
    if argname != "label_scale":
      label_scale = sampler.sample_with_kwargs.keywords["label_scale"]
    self.reg_coef = label_scale * (sqrt(ntasks + high_dim ** 2 / task_size)
                                   / sqrt(ntasks * task_size))

  def fit(self, features, labels):
    ntasks, _, high_dim = features.shape
    loss = partial(nuclear_norm_loss, reg_coef=self.reg_coef)
    grad_fn = jax.grad(loss)
    init = np.random.randn(ntasks * high_dim)
    self.optim_res = optimize.minimize(
      loss, init, args=(features, labels),
      method="L-BFGS-B", jac=lambda *args: np.array(grad_fn(*args)))
    self.weights = np.reshape(self.optim_res.x, (ntasks, high_dim))
    self.subspace = nla.svd(self.weights)[2][:self.low_dim].T
    self.adapt(features, labels)

  def score(self, features, labels):
    """ Computes score of the fit estimator. """
    return -mean_squared_error(features, labels, self.subspace, self.params)


class TransposeNucNorm(NucNorm):
  """ NucNorm estimator which fits on transpose of features, labels. """
  def __init__(self, low_dim, reg_coef=None, axes=(1, 0, 2)):
    super().__init__(low_dim, reg_coef)
    self.axes = axes

  def transpose(self, features, labels):
    """ Transposes features and labels. """
    return (np.transpose(features, self.axes),
            np.transpose(labels, self.axes[:-1]))

  def fit(self, features, labels):
    features, labels = self.transpose(features, labels)
    return super().fit(features, labels)

  def score(self, features, labels):
    features, labels = self.transpose(features, labels)
    return super().score(features, labels)


class NucNormCV(MetaLearningAlg):
  """ Nuclear norm regularized least squares with reg_coef chosen by CV. """
  def __init__(self, low_dim, search_space=None, num_folds=5):
    super().__init__()
    if search_space is None:
      search_space = np.logspace(np.log10(1e-3), np.log10(5), num=10, base=10)
    self.grid_search_cv = GridSearchCV(TransposeNucNorm(low_dim),
                                       {"reg_coef": search_space},
                                       cv=num_folds)
    self.weights = None

  def fit(self, features, labels):
    features = np.transpose(features, (1, 0, 2))
    labels = np.transpose(labels, (1, 0))
    self.grid_search_cv.fit(features, labels)
    self.weights = self.grid_search_cv.best_estimator_.weights
    self.subspace = self.grid_search_cv.best_estimator_.subspace
    self.params = self.grid_search_cv.best_estimator_.params


class NucNormVal(NucNorm):
  """ Nuclear norm regularized alg with reg_coef chosen on validation set. """
  def __init__(self, low_dim, search_space=None, val_size=0.2):
    super().__init__(low_dim)
    if search_space is None:
      search_space = np.logspace(np.log10(1e-3), np.log10(5), num=10, base=10)
    self.search_space = search_space
    self.val_size = val_size
    self.weights = None

  def fit(self, features, labels):
    def transpose(features, labels):
      return np.transpose(features, (1, 0, 2)), np.transpose(labels, (1, 0))

    train_features, val_features, train_labels, val_labels = \
        train_test_split(*transpose(features, labels), test_size=self.val_size)
    train_features, train_labels = transpose(train_features, train_labels)
    val_features, val_labels = transpose(val_features, val_labels)

    best_score = -float("inf")
    best_reg_coef = None
    if self.reg_coef is not None:
      super().fit(train_features, train_labels)
      best_score = super().score(val_features, val_labels)
      best_reg_coef = self.reg_coef
    for reg_coef in self.search_space:
      self.set_params(reg_coef=reg_coef)
      super().fit(train_features, train_labels)
      if (score := super().score(val_features, val_labels)) > best_score:
        best_score = score
        best_reg_coef = reg_coef

    self.set_params(reg_coef=best_reg_coef)
    super().fit(features, labels)


def burer_monteiro_loss(weights, features, labels, low_dim):
  """ Burer-Monteiro loss function. """
  ntasks, _, high_dim = features.shape
  subspace = jnp.reshape(weights[:high_dim * low_dim], (high_dim, low_dim))
  params = jnp.reshape(weights[high_dim * low_dim:], (ntasks, low_dim))
  preds = jnp.sum(features @ subspace * params[:, None], -1)
  prediction_error = 0.5 * jnp.sum(jnp.mean(jnp.square(labels - preds), 1), 0)
  return (prediction_error
          + 1 / 8 * jnp.sum(jnp.square(
            subspace.T @ subspace - params.T @ params)))


class BurerMonteiro(MetaLearningAlg):
  """ Meta-learning via Burer-Monteiro loss optimization. """
  def __init__(self, low_dim):
    super().__init__()
    self.low_dim = low_dim
    self.optim_res = None

  def fit(self, features, labels):
    loss = partial(burer_monteiro_loss, features=features,
                   labels=labels, low_dim=self.low_dim)
    grad_fn = jax.grad(loss)
    ntasks, _, high_dim = features.shape
    init = np.random.normal(size=(high_dim * self.low_dim
                                  + ntasks * self.low_dim))
    self.optim_res = optimize.minimize(
      loss, init, method="L-BFGS-B",
      jac=lambda *args: np.array(grad_fn(*args)))
    self.subspace = nla.svd(
      np.reshape(self.optim_res.x[:high_dim * self.low_dim],
                 (high_dim, self.low_dim))
    )[0][:, :self.low_dim]


class SingleTask(MetaLearningAlg):
  """ Separate regression for each task. """
  def __init__(self, low_dim):
    super().__init__()
    self.weights = None
    self.low_dim = low_dim

  def fit(self, features, labels):
    self.weights = np.asarray([nla.lstsq(X, y, rcond=None)[0]
                               for X, y in zip(features, labels)])
    self.subspace = nla.svd(self.weights)[2][:self.low_dim].T

  def adapt(self, features, labels):
    pass


class LassoSingleTask(MetaLearningAlg):
  """ Single task fit with lasso regularization. """
  def __init__(self, low_dim, fit_intercept=False, **lasso_cv_kwargs):
    super().__init__()
    self.low_dim = low_dim
    self.lasso = LassoCV(fit_intercept=fit_intercept, **lasso_cv_kwargs)
    self.weights = None

  def fit(self, features, labels):
    self.weights = np.asarray([self.lasso.fit(X, y).coef_
                               for X, y in zip(features, labels)])
    self.subspace = nla.svd(self.weights)[2][:self.low_dim].T


class OracleSingleTask(MetaLearningAlg):
  """ Single task learning with oracle subspace. """
  def set_hyperparams(self, sampler, sample, argname, val):
    """ Sets subspace based on the sample. """
    _ = sampler, argname, val
    self.subspace = sample.subspace

  def fit(self, features, labels):
    pass


def random_projection(nrows, ncols):
  """ Returns random projection matrix with the specified dimensions. """
  if nrows <= ncols:
    raise ValueError(f"expected nrows={nrows} to be larger than ncols={ncols}")
  return nla.svd(np.random.normal(size=(nrows, ncols)))[0][:, :ncols]


class RandomProjection(MetaLearningAlg):
  """ Single task learning with random projections. """
  def __init__(self, low_dim):
    super().__init__()
    self.low_dim = low_dim

  def fit(self, features, labels):
    self.subspace = random_projection(features.shape[-1], self.low_dim)
