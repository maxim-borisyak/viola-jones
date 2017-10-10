import numpy as np

from .search import gss


def cumulative(images):
  X = np.ndarray(images.shape, dtype='float32')
  np.cumsum(images, axis=1, out=X)
  np.cumsum(X, axis=2, out=X)

  return X

def _convolve(cum_images, filters):
  return (
    cum_images[:, filters[:, 1, 0], filters[:, 1, 1]] -
    cum_images[:, filters[:, 0, 0], filters[:, 1, 1]] -
    cum_images[:, filters[:, 1, 0], filters[:, 0, 1]] +
    cum_images[:, filters[:, 0, 0], filters[:, 0, 1]]
  )

def _error(y, convs, thresholds, weights):
  predictions = np.where(convs > thresholds[None, :], 1, -1)

  ### mean of weights of miss-classified samples
  error = np.mean(
    (1 - y[:, None] * predictions) * weights[:, None] / 2, axis=0
  )

  return error

def optimal_alpha(y, convs, thresholds, weights):
  predictions = np.where(convs > thresholds[None, :], 1, -1)
  error = np.mean(
    (1 - y[:, None] * predictions) * weights[:, None] / 2, axis=0
  )

  correct = np.mean(
    (1 + y[:, None] * predictions) * weights[:, None] / 2,
    axis=0
  )

  return 0.5 * (np.log(correct) - np.log(error)), error


def _predict_ensemble(images, filters, thresholds, alphas):
  conv = _convolve(images, filters)
  return np.sum(
    np.where(conv > thresholds[None, :], 1, -1) * alphas[None, :],
    axis=1
  )


def _stage_predict_ensemble(images, filters, thresholds, alphas):
  conv = _convolve(images, filters)
  return np.cumsum(
    np.where(conv > thresholds[None, :], 1, -1) * alphas[None, :],
    axis=1
  )

class LightViolaJonesBoost(object):
  def __init__(self,
               n_filters=1024, batch_size=128,
               image_width=28, image_height=28,
               activation_range=(-350.0, 350.0),
               subsampling=None,
               threshold_search_steps=16,
               annealing_rounds=8,
               tweak_filter_eps=1
               ):
    """
    Model for training Haar filters for Viola-Jones cascades.

    The model is, essentially, AdaBoost on Haar filters.
    Instead, of evaluating each possible filter on each stage,
    this implementation samples `batch_size` filters,
    then performs `annealing_rounds` rounds of simplistic simulated annealing-alike search
    on each filter in the batch by randomly changing coordinates of filters
    by +- `tweak_filter_eps` and comparing new 'mutated' filters against current best.

    :param n_filters: number of filters in ensemble.
    :param batch_size: number of filters to sample at each stage.
    :param image_width: size of the first spatial axis;
    :param image_height: size of the second spatial axis;
    :param activation_range: initial search range for filter thresholds.
    :param subsampling: perform sample sampling while evaluated filters.
      If None uses all samples.
      Computations of ensemble weights for base classifiers is done on the whole dataset.
    :param threshold_search_steps: number of golden section search steps for thresholds.
    :param annealing_rounds: number of rounds for the simplistic annealing-alike search.
      If None, annealing is not performed.
    :param tweak_filter_eps: size of the changes in filter coordinates for annealing-alike search.
      Changes are uniformly sampled from `[-tweak_filter_eps, +tweak_filter_eps]`.
    """
    self.activation_range = activation_range
    self.n_filters = n_filters
    self.batch_size = batch_size
    self.image_width = image_width
    self.image_height = image_height

    self.subsampling = subsampling
    self.annealing_rounds = annealing_rounds
    self.threshold_search_steps = threshold_search_steps
    self.tweak_filter_eps = tweak_filter_eps

    self.filters = np.ndarray(
      shape=(n_filters, 2, 2), dtype='int32'
    )

    self.thresholds = np.ndarray(
      shape=(n_filters,), dtype='float32'
    )

    self.alphas = np.ndarray(
      shape=(n_filters,), dtype='float32'
    )

  def save(self, path):
    np.savez(path, filters=self.filters, thresholds=self.thresholds, alphas=self.alphas)

  def load(self, path):
    f = np.load(path)
    self.filters = f['filters'].astype('int32')
    self.thresholds = f['thresholds'].astype('float32')
    self.alphas = f['alphas'].astype('float32')

  def _predict(self, X_cum, filters, thresholds, alphas):
    return _predict_ensemble(X_cum, filters, thresholds, alphas)

  def predict(self, X):
    X_ = cumulative(X)
    return _predict_ensemble(X_, self.filters, self.thresholds, self.alphas)

  def stage_predict(self, X):
    X_ = cumulative(X)
    return _stage_predict_ensemble(X_, self.filters, self.thresholds, self.alphas)

  def get_random_filters(self):
    filters = np.ndarray(shape=(self.batch_size, 2, 2), dtype='int32')
    filters[:, :, 0] = np.random.randint(self.image_width, size=(self.batch_size, 2))
    filters[:, :, 1] = np.random.randint(self.image_height, size=(self.batch_size, 2))
    return filters

  def tweak_filters(self, filters):
    filters = filters + np.random.randint(-1, 2, size=filters.shape)
    filters[:, :, 0] = np.maximum(0, filters[:, :, 0])
    filters[:, :, 0] = np.minimum(self.image_width - 1, filters[:, :, 0])

    filters[:, :, 1] = np.maximum(0, filters[:, :, 1])
    filters[:, :, 1] = np.minimum(self.image_height - 1, filters[:, :, 1])

    return filters.astype('int32')

  def eval_filters(self, candidates, X, y, weights):
    n = X.shape[0]

    if self.subsampling is None:
      convs = _convolve(X, candidates)
    else:
      convs = None

    if self.subsampling is not None:
      indx = np.random.choice(n, size=int(n * self.subsampling)).astype('int32')
      convs = _convolve(X[indx], candidates)
      e = lambda thr: _error(y[indx], convs, thr, weights[indx])
    else:
      e = lambda thr: _error(y, convs, thr, weights)

    ### note that e might be non-convex,
    ### but if it is, the filter is a bad one a priori.
    thresholds, errors = gss(
      e, steps=16, batch_size=candidates.shape[0],
      initial_range=self.activation_range
    )

    return thresholds, errors

  def select_best_filter(self, X, y, weights, n_rounds=32):
    best_filters = self.get_random_filters()
    best_thresholds, min_errors = self.eval_filters(best_filters, X, y, weights)

    for i in range(n_rounds - 1):
      candidates = self.tweak_filters(best_filters)
      thresholds, errors = self.eval_filters(candidates, X, y, weights)

      for j in range(thresholds.shape[0]):
        if errors[j] < min_errors[j]:
          best_filters[j] = candidates[j]
          best_thresholds[j] = thresholds[j]
          min_errors[j] = errors[j]

    convs = _convolve(X, best_filters)
    alphas, errors = optimal_alpha(y, convs, best_thresholds, weights)

    best_of_the_best = np.argmin(errors)

    return best_filters[best_of_the_best], best_thresholds[best_of_the_best], alphas[best_of_the_best]

  def train(self, X, y):
    X_ = cumulative(X)

    weights = np.ones(shape=(X.shape[0]), dtype='float32')

    for i in range(self.n_filters):
      self.filters[i], self.thresholds[i], self.alphas[i] = self.select_best_filter(X_, y,weights, n_rounds=self.annealing_rounds)
      predictions = self._predict(X_, self.filters[:(i + 1)], self.thresholds[:(i + 1)], self.alphas[:(i + 1)])
      losses = np.exp(-y * predictions)
      weights = losses

      yield losses