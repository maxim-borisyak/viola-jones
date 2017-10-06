import numpy as np

from gss import gss

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

  return 0.5 * (np.log(correct) - np.log(error))


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
               activation_range=(-350.0, 350.0)):
    self.activation_range = activation_range
    self.n_filters = n_filters
    self.batch_size = batch_size
    self.image_width = image_width
    self.image_height = image_height

    self.filters = np.ndarray(
      shape=(n_filters, 2, 2), dtype='int32'
    )

    self.thresholds = np.ndarray(
      shape=(n_filters,), dtype='float32'
    )

    self.alphas = np.ndarray(
      shape=(n_filters,), dtype='float32'
    )

    self.X = None

  def set_images(self, X):
    self.X = np.ndarray(X.shape, dtype='float32')
    np.cumsum(X, axis=1, out=self.X)
    np.cumsum(self.X, axis=2, out=self.X)

  def save(self, path):
    np.savez(path, filters=self.filters, thresholds=self.thresholds, alphas=self.alphas)

  def load(self, path):
    f = np.load(path)
    self.filters = f['filters']
    self.thresholds = f['filters']
    self.alphas = f['filters']


  def predict(self, X):
    self.set_images(X)
    return self._predict(self.filters, self.thresholds, self.alphas)

  def stage_predict(self, X):
    self.set_images(X)
    return self._stage_predict(self.filters, self.thresholds, self.alphas)

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

  def eval_filters(self, candidates):
    self.initialize_search(candidates)
    n = self.X_acc.get_value(borrow=True).shape[0]
    for j in range(16):
      self.update_thresholds()

    thresholds = self.get_thresholds()
    errors = self.get_errors()
    alphas = self.get_alpha()

    return thresholds, errors, alphas

  def select_best_filter(self, n_rounds=32):
    best_filters = self.get_random_filters()
    best_thresholds, min_errors, best_alphas = self.eval_filters(best_filters)

    for i in range(n_rounds - 1):
      candidates = self.tweak_filters(best_filters)
      thresholds, errors, alphas = self.eval_filters(candidates)

      for j in range(thresholds.shape[0]):
        if errors[j] < min_errors[j]:
          best_filters[j] = candidates[j]
          best_thresholds[j] = thresholds[j]
          min_errors[j] = errors[j]
          best_alphas[j] = alphas[j]

    best = np.argmin(min_errors)
    return best_filters[best], best_thresholds[best], best_alphas[best]

  def train(self, X, y):
    self.set_images(X)
    self.set_labels(y)

    for i in range(self.n_filters):
      if i == 0:
        self.set_weights(
          np.ones(shape=(X.shape[0]), dtype='float32')
        )
      else:
        predictions = self._predict(self.filters[:i], self.thresholds[:i], self.alphas[:i])
        losses = np.exp(-y * predictions)
        self.set_weights(losses)
        yield losses


      self.filters[i], self.thresholds[i], self.alphas[i] = self.select_best_filter()