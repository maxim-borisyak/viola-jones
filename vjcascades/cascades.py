import numpy as np

import theano
import theano.tensor as T

from gss import golden_section_search

def _convolve(images, filters):
  return (
           images[:, filters[:, 1, 0], filters[:, 1, 1]] -
           images[:, filters[:, 0, 0], filters[:, 1, 1]] -
           images[:, filters[:, 1, 0], filters[:, 0, 1]] +
           images[:, filters[:, 0, 0], filters[:, 0, 1]]
         )

def _error(y, convs, thresholds, weights):
  predictions = T.switch(convs > thresholds[None, :], 1, -1)

  ### mean of weights of miss-classified samples
  error = T.mean(
    (1 - y[:, None] * predictions) * weights[:, None] / 2,
    axis=0
  )

  return error

def optimal_alpha(y, convs, thresholds, weights):
  predictions = T.switch(convs > thresholds[None, :], 1, -1)
  error = T.mean(
    (1 - y[:, None] * predictions) * weights[:, None] / 2,
    axis=0
  )

  correct = T.mean(
    (1 + y[:, None] * predictions) * weights[:, None] / 2,
    axis=0
  )

  return 0.5 * (T.log(correct) - T.log(error))


def _predict_ensemble(images, filters, thresholds, alphas):
  conv = _convolve(images, filters)
  return T.sum(
    T.switch(conv > thresholds[None, :], 1, -1) * alphas[None, :],
    axis=1
  )


def _stage_predict_ensemble(images, filters, thresholds, alphas):
  conv = _convolve(images, filters)
  return T.cumsum(
    T.switch(conv > thresholds[None, :], 1, -1) * alphas[None, :],
    axis=1
  )

class ViolaJonesBoost(object):
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

    X_ = T.ftensor3('X_')
    X_acc = theano.shared(np.ndarray(shape=(0, image_width, image_height), dtype='float32'))
    self.X_acc = X_acc

    self.set_images = theano.function(
      inputs=[X_],
      outputs=None,
      updates=[
        (X_acc, T.cumsum(T.cumsum(X_, axis=2), axis=1))
      ]
    )

    filters = T.itensor3('filters')
    thresholds = T.fvector('threshold')
    alphas = T.fvector('alphas')

    predictions = _predict_ensemble(X_acc, filters, thresholds, alphas)
    self._predict = theano.function(
      inputs=[filters, thresholds, alphas],
      outputs=predictions
    )

    stage_predictions = _stage_predict_ensemble(X_acc, filters, thresholds, alphas)
    self._stage_predict = theano.function(
      inputs=[filters, thresholds, alphas],
      outputs=stage_predictions
    )

    ### training

    weights = theano.shared(
      np.ones(shape=(0,), dtype='float32')
    )
    self.weights = weights

    w = T.fvector('weights')
    self.set_weights = theano.function(
      inputs=[w],
      outputs=None,
      updates=[(weights, w)]
    )

    y_ = T.fvector('y')
    y = theano.shared(np.ndarray(shape=(0, ), dtype='float32'))
    self.set_labels = theano.function(
      inputs=[y_],
      outputs=None,
      updates = [(y, y_)]
    )

    conv_shared = theano.shared(np.ndarray(shape=(0, n_filters), dtype='float32'))
    self.conv_shared = conv_shared

    e = lambda thr: _error(y, conv_shared, thr, weights)
    init_search, upd_search, f, solution = golden_section_search(e, batch_size=batch_size, initial_range=activation_range)

    self.initialize_search = theano.function(
      inputs=[filters],
      outputs=None,
      updates=[
        (conv_shared, _convolve(X_acc, filters))
      ] + init_search
    )

    self.update_thresholds = theano.function(
      inputs=[],
      outputs=None,
      updates=upd_search
    )

    self.get_thresholds = theano.function(inputs=[], outputs=solution)
    self.get_errors = theano.function(inputs=[], outputs=f)

    self.get_alpha = theano.function(
      inputs=[],
      outputs=optimal_alpha(y, conv_shared, solution, weights)
    )

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
    for j in range(64):
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