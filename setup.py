"""
  Viola-Jones cascades.
"""

from setuptools import setup, find_packages, Extension

from codecs import open
import os
import os.path as osp
import numpy as np

here = osp.abspath(osp.dirname(__file__))

with open(osp.join(here, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

extensions = [
]

setup(
  name='vjcascades',

  version='1.0.0',

  description="""Viola-Jones cascades implemented on theano""",

  long_description = long_description,

  url='https://github.com/maxim-borisyak/viola-jones',

  author='Maxim Borisyak',
  author_email='mborisyak at hse dot ru',

  maintainer = 'Maxim Borisyak',
  maintainer_email = 'mborisyak at hse dot ru',

  license='MIT',

  classifiers=[
    'Development Status :: 4 - Beta',

    'Intended Audience :: Science/Research',

    'Topic :: Scientific/Engineering :: Physics',

    'License :: OSI Approved :: MIT License',

    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.6',
    'Programming Language :: Python :: 2.7',
  ],

  keywords='Viola-Jones',

  packages=find_packages(exclude=['contrib', 'examples', 'docs', 'tests']),

  extras_require={
    'dev': ['check-manifest'],
    'test': ['nose>=1.3.0'],
  },

  install_requires=[
    'numpy',
  ],

  include_package_data=True,

  package_data = {
  },
)