import numpy as np

import theano
import theano.tensor as T

__all__ = [
  'golden_section_search',
  'gss'
]

def golden_section_search(f, batch_size=128, initial_range=(0, 1)):
  """This implementation could be easily speeded up."""
  golden_ratio = T.constant((np.sqrt(5) + 1) / 2, dtype='float32')

  left = theano.shared(np.zeros(shape=(batch_size,), dtype='float32'))
  right = theano.shared(np.zeros(shape=(batch_size,), dtype='float32'))

  delta = (right - left) / golden_ratio

  middle = right - delta
  prob = left + delta

  middle_f = f(middle)
  prob_f = f(prob)

  a, b = initial_range

  init = [
    (left, T.ones(shape=batch_size) * a),
    (right, T.ones(shape=batch_size) * b)
  ]

  shift_left = prob_f > middle_f

  upd = [
    (left, T.switch(
      shift_left,
      left,
      middle
    )),
    (right, T.switch(
      shift_left,
      prob,
      right
    ))
  ]

  return init, upd, middle_f, middle


def gss(f, steps=64, batch_size=128, initial_range=(0, 1)):
  """This implementation could be easily speeded up."""
  golden_ratio = (np.sqrt(5) + 1) / 2

  a, b = initial_range

  left = np.ones(shape=(batch_size,), dtype='float32') * a
  right = np.ones(shape=(batch_size,), dtype='float32') * b

  delta = (b - a) / golden_ratio

  middle = np.ones(shape=(batch_size,), dtype='float32') * (b - delta / golden_ratio)
  prob = np.ones(shape=(batch_size,), dtype='float32') * (a + delta / golden_ratio)

  #f_left = f(left)
  #f_right = f(right)
  f_middle = f(middle)
  f_prob = f(prob)

  requests = np.ndarray(shape=(batch_size, ), dtype='float32')

  for i in range(steps):
    shift_left = f_middle < f_prob

    for j in range(batch_size):
      if shift_left[j] > 0:
        right[j] = prob[j]
        #f_right = f_prob[j]
        prob[j] = middle[j]
        f_prob[j] = f_middle[j]
        middle[j] = right[j] - (right[j] - left[j]) / golden_ratio
        requests[j] = middle[j]
      else:
        left[j] = middle[j]
        #f_left[j] = f_middle[j]

        middle[j] = prob[j]
        f_middle[j] = f_prob[j]

        prob[j] = left[j] + (right[j] - left[j]) / golden_ratio
        requests[j] = prob[j]

      requested = f(requests)
      for j in range(batch_size):
        if shift_left[j] > 0:
          f_middle[j] = requested[j]
        else:
          f_prob[j] = requested[j]



  return