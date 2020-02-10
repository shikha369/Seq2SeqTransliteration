from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import random_ops


class Initializer(object):
  """Initializer base class: all initializers inherit from this class.
  """

  def __call__(self, shape, dtype=None, partition_info=None):
    raise NotImplementedError


class Zeros(Initializer):
  """Initializer that generates tensors initialized to 0."""

  def __init__(self, dtype=dtypes.float32):
    self.dtype = dtype

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return constant_op.constant(False if dtype is dtypes.bool else 0,
                                dtype=dtype, shape=shape)


class Ones(Initializer):
  """Initializer that generates tensors initialized to 1."""

  def __init__(self, dtype=dtypes.float32):
    self.dtype = dtype

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return constant_op.constant(1, dtype=dtype, shape=shape)


class Constant(Initializer):
  
  def __init__(self, value=0, dtype=dtypes.float32):
    self.value = value
    self.dtype = dtype

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return constant_op.constant(self.value, dtype=dtype, shape=shape)


class RandomUniform(Initializer):
  
  def __init__(self, minval=0, maxval=None, seed=None, dtype=dtypes.float32):
    self.minval = minval
    self.maxval = maxval
    self.seed = seed
    self.dtype = dtype

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return random_ops.random_uniform(shape, self.minval, self.maxval,
                                     dtype, seed=self.seed)


class RandomNormal(Initializer):
  

  def __init__(self, mean=0.0, stddev=1.0, seed=None, dtype=dtypes.float32):
    self.mean = mean
    self.stddev = stddev
    self.seed = seed
    self.dtype = _assert_float_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return random_ops.random_normal(shape, self.mean, self.stddev,
                                    dtype, seed=self.seed)


class TruncatedNormal(Initializer):
  
  def __init__(self, mean=0.0, stddev=1.0, seed=None, dtype=dtypes.float32):
    self.mean = mean
    self.stddev = stddev
    self.seed = seed
    self.dtype = _assert_float_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return random_ops.truncated_normal(shape, self.mean, self.stddev,
                                       dtype, seed=self.seed)


class UniformUnitScaling(Initializer):
  

  def __init__(self, factor=1.0, seed=None, dtype=dtypes.float32):
    self.factor = factor
    self.seed = seed
    self.dtype = _assert_float_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    scale_shape = shape
    if partition_info is not None:
      scale_shape = partition_info.full_shape

    input_size = 1.0
    # Estimating input size is not possible to do perfectly, but we try.
    # The estimate, obtained by multiplying all dimensions but the last one,
    # is the right thing for matrix multiply and convolutions (see above).
    for dim in scale_shape[:-1]:
      input_size *= float(dim)
    # Avoid errors when initializing zero-size tensors.
    input_size = max(input_size, 1.0)
    max_val = math.sqrt(3 / input_size) * self.factor
    return random_ops.random_uniform(shape, -max_val, max_val,
                                     dtype, seed=self.seed)


class VarianceScaling(Initializer):
  
  def __init__(self, scale=1.0,
               mode="fan_in",
               distribution="normal",
               seed=None,
               dtype=dtypes.float32):
    if scale <= 0.:
      raise ValueError("`scale` must be positive float.")
    if mode not in {"fan_in", "fan_out", "fan_avg"}:
      raise ValueError("Invalid `mode` argument:", mode)
    distribution = distribution.lower()
    if distribution not in {"normal", "uniform"}:
      raise ValueError("Invalid `distribution` argument:", distribution)
    self.scale = scale
    self.mode = mode
    self.distribution = distribution
    self.seed = seed
    self.dtype = _assert_float_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    scale = self.scale
    scale_shape = shape
    if partition_info is not None:
      scale_shape = partition_info.full_shape
    fan_in, fan_out = _compute_fans(scale_shape)
    if self.mode == "fan_in":
      scale /= max(1., fan_in)
    elif self.mode == "fan_out":
      scale /= max(1., fan_out)
    else:
      scale /= max(1., (fan_in + fan_out) / 2.)
    if self.distribution == "normal":
      stddev = math.sqrt(scale)
      return random_ops.truncated_normal(shape, 0.0, stddev,
                                         dtype, seed=self.seed)
    else:
      limit = math.sqrt(3.0 * scale)
      return random_ops.random_uniform(shape, -limit, limit,
                                       dtype, seed=self.seed)


class Orthogonal(Initializer):
  
  def __init__(self, gain=1.0, dtype=dtypes.float32, seed=None):
    self.gain = gain
    self.dtype = _assert_float_dtype(dtype)
    self.seed = seed

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    # Check the shape
    if len(shape) < 2:
      raise ValueError("The tensor to initialize must be "
                       "at least two-dimensional")
    # Flatten the input shape with the last dimension remaining
    # its original shape so it works for conv2d
    num_rows = 1
    for dim in shape[:-1]:
      num_rows *= dim
    num_cols = shape[-1]
    flat_shape = (num_rows, num_cols)

    # Generate a random matrix
    a = random_ops.random_uniform(flat_shape, dtype=dtype, seed=self.seed)
    # Compute the svd
    _, u, v = linalg_ops.svd(a, full_matrices=False)
    # Pick the appropriate singular value decomposition
    if num_rows > num_cols:
      q = u
    else:
      # Tensorflow departs from numpy conventions
      # such that we need to transpose axes here
      q = array_ops.transpose(v)
    return self.gain * array_ops.reshape(q, shape)


# Aliases.

# pylint: disable=invalid-name
zeros_initializer = Zeros
ones_initializer = Ones
constant_initializer = Constant
random_uniform_initializer = RandomUniform
random_normal_initializer = RandomNormal
truncated_normal_initializer = TruncatedNormal
uniform_unit_scaling_initializer = UniformUnitScaling
variance_scaling_initializer = VarianceScaling
orthogonal_initializer = Orthogonal
# pylint: enable=invalid-name


def glorot_uniform_initializer(seed=None, dtype=dtypes.float32):
  return variance_scaling_initializer(scale=1.0,
                                      mode="fan_avg",
                                      distribution="uniform",
                                      seed=seed,
                                      dtype=dtype)


def glorot_normal_initializer(seed=None, dtype=dtypes.float32):
  return variance_scaling_initializer(scale=1.0,
                                      mode="fan_avg",
                                      distribution="normal",
                                      seed=seed,
                                      dtype=dtype)


# Utility functions.


def _compute_fans(shape):
  if len(shape) < 1:  # Just to avoid errors for constants.
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  else:
    # Assuming convolution kernels (2D, 3D, or more).
    # kernel shape: (..., input_depth, depth)
    receptive_field_size = 1.
    for dim in shape[:-2]:
      receptive_field_size *= dim
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  return fan_in, fan_out


def _assert_float_dtype(dtype):
  if not dtype.is_floating:
    raise ValueError("Expected floating point type, got %s." % dtype)
  return dtype
