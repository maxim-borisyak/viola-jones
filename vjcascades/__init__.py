try:
  from .theanoimpl import ViolaJonesBoost
except ImportError:
  pass

from .numpyimpl import LightViolaJonesBoost