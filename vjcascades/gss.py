import numpy as np

import theano
import theano.tensor as T

__all__ = [
  'golden_section_search'
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