from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.util.deprecation import deprecated

def normalization(scores):
  return scores-tf.tile(tf.transpose([tf.reduce_mean(scores,1)]),[1,tf.cast(scores.get_shape()[1], dtype=tf.int32)])

def add_loss(loss, loss_collection=ops.GraphKeys.LOSSES):
  """Adds a externally defined loss to the collection of losses.

  Args:
    loss: A loss `Tensor`.
    loss_collection: Optional collection to add the loss to.
  """
  if loss_collection:
    ops.add_to_collection(loss_collection, loss)

def _scale_losses(losses, weights):
  """Computes the scaled loss.

  Args:
    losses: A `Tensor` of size [batch_size, d1, ... dN].
    weights: A `Tensor` of size [1], [batch_size] or [batch_size, d1, ... dN].
      The `losses` are reduced (tf.reduce_sum) until its dimension matches
      that of `weights` at which point the reduced `losses` are element-wise
      multiplied by `weights` and a final reduce_sum is computed on the result.
      Conceptually, this operation is equivalent to broadcasting (tiling)
      `weights` to be the same size as `losses`, performing an element-wise
      multiplication, and summing the result.

  Returns:
    A scalar tf.float32 `Tensor` whose value represents the sum of the scaled
      `losses`.
  """
  # First, compute the sum of the losses over all elements:
  start_index = max(0, weights.get_shape().ndims)
  reduction_indices = list(range(start_index, losses.get_shape().ndims))
  reduced_losses = math_ops.reduce_sum(losses,
                                       reduction_indices=reduction_indices)
  reduced_losses = math_ops.multiply(reduced_losses, weights)
  return math_ops.reduce_sum(reduced_losses)


def _safe_div(numerator, denominator, name="value"):
  """Computes a safe divide which returns 0 if the denominator is zero.

  Note that the function contains an additional conditional check that is
  necessary for avoiding situations where the loss is zero causing NaNs to
  creep into the gradient computation.

  Args:
    numerator: An arbitrary `Tensor`.
    denominator: A `Tensor` whose shape matches `numerator` and whose values are
      assumed to be non-negative.
    name: An optional name for the returned op.

  Returns:
    The element-wise value of the numerator divided by the denominator.
  """
  return array_ops.where(
      math_ops.greater(denominator, 0),
      math_ops.div(numerator, array_ops.where(
          math_ops.equal(denominator, 0),
          array_ops.ones_like(denominator), denominator)),
      array_ops.zeros_like(numerator),
      name=name)


def _safe_mean(losses, num_present):
  """Computes a safe mean of the losses.

  Args:
    losses: A tensor whose elements contain individual loss measurements.
    num_present: The number of measurable losses in the tensor.

  Returns:
    A scalar representing the mean of the losses. If `num_present` is zero,
      then zero is returned.
  """
  total_loss = math_ops.reduce_sum(losses)
  return _safe_div(total_loss, num_present)


@deprecated("2016-12-30", "Use tf.losses.compute_weighted_loss instead.")
def compute_weighted_loss(losses, weights=1.0, scope=None):
  """Computes the weighted loss.

  Args:
    losses: A tensor of size [batch_size, d1, ... dN].
    weights: A tensor of size [1] or [batch_size, d1, ... dK] where K < N.
    scope: the scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` that returns the weighted loss.

  Raises:
    ValueError: If `weights` is `None` or the shape is not compatible with
      `losses`, or if the number of dimensions (rank) of either `losses` or
      `weights` is missing.
  """
  with ops.name_scope(scope, "weighted_loss", [losses, weights]):
    losses = ops.convert_to_tensor(losses)
    input_dtype = losses.dtype
    losses = math_ops.to_float(losses)
    weights = math_ops.to_float(ops.convert_to_tensor(weights))

    if losses.get_shape().ndims is None:
      raise ValueError("losses.get_shape().ndims cannot be None")
    weights_shape = weights.get_shape()
    if weights_shape.ndims is None:
      raise ValueError("weights.get_shape().ndims cannot be None")

    if weights_shape.ndims > 1 and weights_shape.dims[-1].is_compatible_with(1):
      weights = array_ops.squeeze(weights, [-1])

    total_loss = _scale_losses(losses, weights)
    num_present = _num_present(losses, weights)
    mean_loss = _safe_mean(total_loss, num_present)
    # convert the result back to the input type
    mean_loss = math_ops.cast(mean_loss, input_dtype)
    add_loss(mean_loss)
    return mean_loss


def _num_present(losses, weights, per_batch=False):
  """Computes the number of elements in the loss function induced by `weights`.

  A given weights tensor induces different numbers of usable elements in the
  `losses` tensor. The `weights` tensor is broadcast across `losses` for all
  possible dimensions. For example, if `losses` is a tensor of dimension
  [4, 5, 6, 3] and `weights` is a tensor of size [4, 5], then `weights` is, in
  effect, tiled to match the size of `losses`. Following this effective tile,
  the total number of present elements is the number of non-zero weights.

  Args:
    losses: A tensor of size [batch_size, d1, ... dN].
    weights: A tensor of size [1] or [batch_size, d1, ... dK] where K < N.
    per_batch: Whether to return the number of elements per batch or as a sum
      total.

  Returns:
    The number of present (non-zero) elements in the losses tensor. If
      `per_batch` is True, the value is returned as a tensor of size
      [batch_size]. Otherwise, a single scalar tensor is returned.
  """
  # If weights is a scalar, its easy to compute:
  if weights.get_shape().ndims == 0:
    batch_size = array_ops.reshape(array_ops.slice(array_ops.shape(losses),
                                                   [0], [1]), [])
    num_per_batch = math_ops.div(math_ops.to_float(array_ops.size(losses)),
                                 math_ops.to_float(batch_size))
    num_per_batch = array_ops.where(math_ops.equal(weights, 0),
                                    0.0, num_per_batch)
    num_per_batch = math_ops.multiply(array_ops.ones(
        array_ops.reshape(batch_size, [1])), num_per_batch)
    return num_per_batch if per_batch else math_ops.reduce_sum(num_per_batch)

  # First, count the number of nonzero weights:
  if weights.get_shape().ndims >= 1:
    reduction_indices = list(range(1, weights.get_shape().ndims))
    num_nonzero_per_batch = math_ops.reduce_sum(
        math_ops.to_float(math_ops.not_equal(weights, 0)),
        reduction_indices=reduction_indices)

  # Next, determine the number of elements that weights would broadcast to:
  broadcast_dims = array_ops.slice(array_ops.shape(losses),
                                   [weights.get_shape().ndims], [-1])
  num_to_broadcast = math_ops.to_float(math_ops.reduce_prod(broadcast_dims))

  num_per_batch = math_ops.multiply(num_nonzero_per_batch, num_to_broadcast)
  return num_per_batch if per_batch else math_ops.reduce_sum(num_per_batch)


def multiclasscrammerloss(scores, classes, batch_size, weights, scope=None):
  scores=tf.transpose(normalization(scores))
  classes=tf.transpose(classes)
  true_classes = tf.argmax(classes, 0)
  idx_flattened = tf.range(0, batch_size) * scores.get_shape()[0]+ tf.cast(true_classes, dtype=tf.int32)
  true_scores = tf.gather(tf.reshape(tf.transpose(scores), [-1]),
                          idx_flattened)
  losses=tf.nn.relu(1-true_scores + tf.reduce_max(scores*(1-classes),reduction_indices=[0]))
  #losses=tf.reduce_mean(L)
  return compute_weighted_loss(losses, weights, scope=scope)

def multiclasshingeloss(scores, classes, batch_size, weights, scope=None):
  scores=tf.transpose(normalization(scores))
  classes=tf.transpose(classes)
  true_classes = tf.argmax(classes, 0)
  idx_flattened = tf.range(0, batch_size) * scores.get_shape()[0]+ tf.cast(true_classes, dtype=tf.int32)
  true_scores = tf.gather(tf.reshape(tf.transpose(scores), [-1]),
                          idx_flattened)
  losses = tf.nn.relu((1 - true_scores + scores) * (1 - classes))
  return compute_weighted_loss(losses, weights, scope=scope)

def multiclassleeloss(scores,classes,nb_class, batch_size, weights, scope=None):
  scores=tf.transpose(normalization(scores))
  classes=tf.transpose(classes)
  true_classes = tf.argmax(classes, 0)
  idx_flattened = tf.range(0, batch_size) * scores.get_shape()[0]+ tf.cast(true_classes, dtype=tf.int32)
  true_scores = tf.gather(tf.reshape(tf.transpose(scores), [-1]),
                          idx_flattened)
  L = tf.nn.relu( (1/(nb_class-1) + scores) * (1 - classes))
  losses=tf.reduce_sum(L,0)
  #losses=tf.reduce_mean(l)
  return compute_weighted_loss(losses, weights, scope=scope)

def surrogate_hinge(scores, classes, weights, scope=None):
  scores = tf.transpose(normalization(scores))
  classes = tf.transpose(classes)
  L = tf.nn.relu((1 + scores)*(1-classes))
  losses = tf.reduce_sum(L, 0)
  #final_loss = tf.reduce_mean(l)
  return compute_weighted_loss(losses, weights, scope=scope)

def surrogate_hinge_squares(scores, classes, weights, scope=None):
  scores = tf.transpose(normalization(scores))
  classes = tf.transpose(classes)
  L = tf.nn.relu((1 + scores)*(1-classes))
  losses = tf.reduce_sum(tf.square(L), 0)
  #final_loss = tf.reduce_mean(l)
  return compute_weighted_loss(losses, weights, scope=scope)

def surrogate_squares(scores, classes, weights, scope=None):
  scores = tf.transpose(normalization(scores))
  classes = tf.transpose(classes)
  L = (1 + scores)*(1-classes)
  losses = tf.reduce_sum(tf.square(L), 0)
  #final_loss = tf.reduce_mean(l)
  return compute_weighted_loss(losses, weights, scope=scope)

def surrogate_exponential(scores, classes, weights, scope=None):
  scores = tf.transpose(normalization(scores))
  classes = tf.transpose(classes)
  L = tf.exp(scores)*(1-classes)
  losses = tf.reduce_sum(L, 0)
  #final_loss = tf.reduce_mean(l)
  return compute_weighted_loss(losses, weights, scope=scope)

def surrogate_sigmoid(scores, classes, weights, scope=None):
  scores = tf.transpose(normalization(scores))
  classes = tf.transpose(classes)
  L = (1+tf.tanh(scores))*(1-classes)
  losses = tf.reduce_sum(L, 0)
  #final_loss = tf.reduce_mean(l)
  return compute_weighted_loss(losses, weights, scope=scope)

def surrogate_logistic(scores, classes, weights, scope=None):
  scores = tf.transpose(normalization(scores))
  classes = tf.transpose(classes)
  L = tf.log(1+tf.exp(scores))*(1-classes)
  losses = tf.reduce_sum(L, 0)
  #final_loss = tf.reduce_mean(l)
  return compute_weighted_loss(losses, weights, scope=scope)

def surrogate_double_hinge(scores, classes, weights, scope=None):
  scores = tf.transpose(normalization(scores))
  classes = tf.transpose(classes)
  L = tf.subtract(tf.nn.relu(1 + scores),tf.nn.relu(scores))*(1-classes)
  losses = tf.reduce_sum(L, 0)
  #final_loss = tf.reduce_mean(l)
  return compute_weighted_loss(losses, weights, scope=scope)

def entropy(scores, classes, weights, scope=None):
  losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=classes, logits=scores, name='cross_entropy_per_example')
  return compute_weighted_loss(losses, weights, scope=scope)

def GEL(scores, classes, batch_size, weights, scope=None):
  scores=tf.transpose(normalization(scores))
  #scores=tf.cast(scores, tf.float64)
  classes=tf.transpose(classes)
  true_classes = tf.argmax(classes, 0)
  idx_flattened = tf.range(0, batch_size) * scores.get_shape()[0]+ tf.cast(true_classes, dtype=tf.int32)
  true_scores = tf.gather(tf.reshape(tf.transpose(scores), [-1]),
                          idx_flattened)
  L = tf.exp(scores - true_scores)*(1 - classes)
  losses = tf.reduce_sum(L, 0)
  return compute_weighted_loss(losses, weights, scope=scope)

def GLL(scores, classes, batch_size, weights, scope=None):
  scores=tf.transpose(normalization(scores))
  classes=tf.transpose(classes)
  true_classes = tf.argmax(classes, 0)
  idx_flattened = tf.range(0, batch_size) * scores.get_shape()[0]+ tf.cast(true_classes, dtype=tf.int32)
  true_scores = tf.gather(tf.reshape(tf.transpose(scores), [-1]),
                          idx_flattened)
  L = tf.exp(scores - true_scores)*(1 - classes)
  losses = tf.log1p(tf.reduce_sum(L, 0))
  #final_loss = tf.reduce_mean(l)
  return compute_weighted_loss(losses, weights, scope=scope)

def large_margin_scores(basic_scores,weights):
  weight_norm=tf.sqrt(tf.reduce_sum(tf.square(weights),0))
  x_norm=tf.sqrt(tf.reduce_sum(tf.square(basic_scores),1,keep_dims=True))
  norm=tf.multiply(x_norm,weight_norm)

  scalar_product=tf.matmul(basic_scores,weights)
  cosinus=tf.divide(scalar_product,norm)

  test2=tf.pow(tf.fill(tf.shape(scalar_product),-1.0),tf.floor(1-cosinus))
  oo=2*tf.multiply(tf.square(cosinus),test2)-1
  return tf.multiply(oo,norm)