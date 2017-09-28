import theano
theano.config.floatX = 'float32'

from vjcascades import ViolaJonesBoost

import numpy as np

import os

def test_vjboost():
  boost = ViolaJonesBoost(
    n_filters=16,
    batch_size=8,
    subsampling=0.5
  )

  X = np.random.uniform(0, 1, size=(100, 28, 28)).astype('float32')
  y = np.random.randint(2, size=100)
  y = (2 * y - 1).astype('float32')

  for _ in boost.train(X, y):
    pass
  predictions = boost.predict(X)

  assert predictions.shape == (100, )

  fs, thrs, alphas = boost.filters.copy(), boost.thresholds.copy(), boost.alphas.copy()

  try:
    boost.save('tmp.npz')
    boost.load('tmp.npz')
  finally:
    try:
      os.remove('tmp.npz')
    except:
      pass

  assert np.allclose(boost.filters, fs)
  assert np.allclose(boost.thresholds, thrs)
  assert np.allclose(boost.alphas, alphas)

  assert np.allclose(predictions, boost.predict(X))

  for _ in boost.train(X, y):
    pass
  predictions = boost.predict(X)
  assert predictions.shape == (100, )




