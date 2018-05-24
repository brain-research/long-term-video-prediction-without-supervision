# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convenience functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

from tensorflow.contrib import slim


def shape_initializer(shape, dtype=tf.float32):
  """Adaptor for zeros_initializer, lets shape to be unknown at compile time.

  Args:
    shape: Either a list of integers, or a 1-D Tensor of type int32.
    dtype: The type of an element in the resulting Tensor.

  Returns:
    A Tensor with all elements set to zero.
  """
  return tf.zeros(shape, dtype)


def init_state(inputs,
               state_shape,
               state_initializer=shape_initializer,
               dtype=tf.float32):
  """Helper function to create an initial state given inputs.

  Args:
    inputs: input Tensor, at least 2D, the first dimension being batch_size
    state_shape: the shape of the state.
    state_initializer: Initializer(shape, dtype) for state Tensor.
    dtype: Optional dtype, needed when inputs is None.
  Returns:
     A tensors representing the initial state.
  """
  if inputs is not None:
    # Handle both the dynamic shape as well as the inferred shape.
    inferred_batch_size = inputs.get_shape().with_rank_at_least(1)[0]
    batch_size = tf.shape(inputs)[0]
    dtype = inputs.dtype
  else:
    inferred_batch_size = 0
    batch_size = 0

  initial_state = state_initializer(
      tf.stack([batch_size] + state_shape), dtype=dtype)
  initial_state.set_shape([inferred_batch_size] + state_shape)

  return initial_state


def _get_concat_variable(name, shape, dtype, num_shards):
  """Get a sharded variable concatenated into one tensor."""
  sharded_variable = _get_shared_variable(name, shape, dtype, num_shards)
  if len(sharded_variable) == 1:
    return sharded_variable[0]

  concat_name = name + '/concat'
  concat_full_name = tf.get_variable_scope().name + '/' + concat_name + ':0'
  for value in tf.get_collection(tf.GraphKeys.CONCATENATED_VARIABLES):
    if value.name == concat_full_name:
      return value

  concat_variable = tf.concat(sharded_variable, 0, name=concat_name)
  tf.add_to_collection(tf.GraphKeys.CONCATENATED_VARIABLES, concat_variable)
  return concat_variable


def _get_shared_variable(name, shape, dtype, num_shards):
  """Get a list of sharded variables with the given dtype."""
  if num_shards > shape[0]:
    raise ValueError(
        'Too many shards: shape=%s, num_shards=%d' % (shape, num_shards))
  unit_shard_size = int(math.floor(shape[0] / num_shards))
  remaining_rows = shape[0] - unit_shard_size * num_shards

  shards = []
  for i in range(num_shards):
    current_size = unit_shard_size
    if i < remaining_rows:
      current_size += 1
    shard = tf.contrib.framework.variable(
        name + '_%d' % i, shape=[current_size, shape[1]], dtype=dtype)
    shards.append(shard)
  return shards


def concat(inputs, state):
  """Helper function to concatenate inputs and state along dim=1.

  Args:
    inputs: input Tensor, 2D, batch_size x input_size.
    state: state Tensor, 2D, batch_size x state_size.
  Returns:
     a concatenated Tensor, 2D, batch_size x (input_size + state_size).
  """
  return tf.concat([inputs, state], 1)


@tf.contrib.framework.add_arg_scope
def lstm_cell(inputs,
              state,
              num_units,
              use_peepholes=False,
              cell_clip=0.0,
              initializer=None,
              num_proj=None,
              num_unit_shards=1,
              num_proj_shards=1,
              scope=None,
              reuse=None):
  """Long short-term memory unit (LSTM) recurrent network cell.

  This implementation is based on:

  https://research.google.com/pubs/archive/43905.pdf

  Hasim Sak, Andrew Senior, and Francoise Beaufays.
  "Long short-term memory recurrent neural network architectures for
   large scale acoustic modeling." INTERSPEECH, 2014.

  It uses peep-hole connections, optional cell clipping, and an optional
  projection layer.

  Args:
    inputs: input Tensor, 2D, batch x num_units.
    state: state Tensor, 2D, batch x state_size.
    num_units: int, The number of units in the LSTM cell
    use_peepholes: bool, set True to enable diagonal/peephole connections.
    cell_clip: (optional) A float value, if provided the cell state is clipped
      by this value prior to the cell output activation.
    initializer: (optional) The initializer to use for the weight and
      projection matrices.
    num_proj: (optional) int, The output dimensionality for the projection
      matrices.  If None, no projection is performed.
    num_unit_shards: How to split the weight matrix.  If >1, the weight
      matrix is stored across num_unit_shards.
    num_proj_shards: How to split the projection matrix.  If >1, the
      projection matrix is stored across num_proj_shards.
    scope: Optional scope for variable_scope.
    reuse: whether or not the layer and the variables should be reused.

  Returns:
    A tuple containing:
    - A 2D, batch x output_dim, Tensor representing the output of the LSTM
      after reading "inputs" when previous state was "state".
      Here output_dim is:
         num_proj if num_proj was set,
         num_units otherwise.
    - A 2D, batch x state_size, Tensor representing the new state of LSTM
      after reading "inputs" when previous state was "state".
  """
  num_proj = num_units if num_proj is None else num_proj
  if state is None:
    state = init_state(inputs, [num_units + num_proj])
  with tf.variable_scope(
      scope, 'LSTMCell', [inputs, state], initializer=initializer, reuse=reuse):
    inputs.get_shape().assert_has_rank(2)
    state.get_shape().assert_has_rank(2)
    c_prev = tf.strided_slice(state, [0, 0], [-1, num_units], end_mask=1 << 0)
    m_prev = tf.strided_slice(
        state, [0, num_units], [-1, num_units + num_proj], end_mask=1 << 0)

    dtype = inputs.dtype
    actual_input_size = inputs.get_shape()[1].value

    concat_w = _get_concat_variable(
        'W', [actual_input_size + num_proj, 4 * num_units], dtype,
        num_unit_shards)

    b = tf.contrib.framework.variable(
        'B',
        shape=[4 * num_units],
        dtype=dtype,
        initializer=tf.zeros_initializer())

    cell_inputs = concat(inputs, m_prev)
    lstm_matrix = tf.nn.xw_plus_b(cell_inputs, concat_w, b)
    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = tf.split(value=lstm_matrix, num_or_size_splits=4, axis=1)
    # Diagonal connections
    if use_peepholes:
      w_f_diag = tf.contrib.framework.variable(
          'W_F_diag', shape=[num_units], dtype=inputs.dtype)
      w_i_diag = tf.contrib.framework.variable(
          'W_I_diag', shape=[num_units], dtype=inputs.dtype)
      w_o_diag = tf.contrib.framework.variable(
          'W_O_diag', shape=[num_units], dtype=inputs.dtype)
      c = (
          tf.sigmoid(f + 1 + w_f_diag * c_prev) * c_prev +
          tf.sigmoid(i + w_i_diag * c_prev) * tf.tanh(j))
    else:
      c = (tf.sigmoid(f + 1) * c_prev + tf.sigmoid(i) * tf.tanh(j))

    if cell_clip:
      c = tf.clip_by_value(c, -cell_clip, cell_clip)

    if use_peepholes:
      m = tf.sigmoid(o + w_o_diag * c) * tf.tanh(c)
    else:
      m = tf.sigmoid(o) * tf.tanh(c)

    if num_proj is not None:
      concat_w_proj = _get_concat_variable('W_P', [num_units, num_proj],
                                           inputs.dtype, num_proj_shards)

      m = tf.matmul(m, concat_w_proj)

    return m, concat(c, m)


def get_add_and_check_is_final(end_points, prefix, final_endpoint):

  def add_and_check_is_final(layer_name, net):
    end_points['%s/%s' % (prefix, layer_name)] = net
    return layer_name == final_endpoint

  return add_and_check_is_final


def get_repeat(end_points, prefix, final_endpoint):
  """Simulate `slim.repeat`, and add to endpoints dictionary."""

  def repeat(net, repetitions, layer, *args, **kwargs):
    base_scope = kwargs['scope']
    add_and_check_is_final = get_add_and_check_is_final(end_points, prefix,
                                                        final_endpoint)
    with tf.variable_scope(base_scope, [net]):
      for i in xrange(repetitions):
        kwargs['scope'] = base_scope + '_' + str(i + 1)
        net = layer(net, *args, **kwargs)
        if add_and_check_is_final('%s_%i' % (base_scope, i), net):
          break
      return net

  return repeat


def vgg_arg_scope(weight_decay=0.0005, stddev=0.01):
  trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      activation_fn=tf.nn.relu,
      weights_initializer=trunc_normal(stddev),
      weights_regularizer=slim.l2_regularizer(weight_decay),
      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc


def vgg_16(inputs,
           num_classes=1000,
           dropout_keep_prob=0.5,
           is_training=True,
           predictions_fn=slim.softmax,
           spatial_squeeze=True,
           final_endpoint=None,
           reuse=None,
           scope='vgg_16',
           global_pool=False):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  The default image size used to train this network is 224x224.

  The appropiate weight_decay and stddev are set via the vgg_arg_scope.

  Example:
    with slim.arg_scope(vgg_arg_scope(weight_decay=0.0005)):
      outputs, end_points = vgg_16(inputs)

  Args:
    inputs: a tensor of size [batch_size, height, width, channels]. Must be
      floating point. If a pretrained checkpoint is used, pixel values should be
      the same as during training.
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    dropout_keep_prob: the probability of dropping hidden units during training.
    is_training: whether or not the model is being trained.
    predictions_fn: a function to get predictions out of logits.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'pool3',
      'conv4', 'pool4', 'conv5', 'pool5', 'fc6', 'fc7', 'fc8'], or 'global_pool'
      if that flag is set. By default, the entire network is built.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional scope for the variables.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations. For
    backwards compatibility, some Tensors appear multiple times in the dict.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values.
  """
  if not final_endpoint:
    final_endpoint = ('fc8' if num_classes else 'global_pool'
                      if global_pool else 'fc7')
  end_points = {}

  with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=reuse) as sc:

    add_and_check_is_final = get_add_and_check_is_final(end_points, sc.name,
                                                        final_endpoint)
    repeat = get_repeat(end_points, sc.name, final_endpoint)

    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
      net = repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      if add_and_check_is_final('conv1', net):
        return net, end_points
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      if add_and_check_is_final('pool1', net):
        return net, end_points
      net = repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      if add_and_check_is_final('conv2', net):
        return net, end_points
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      if add_and_check_is_final('pool2', net):
        return net, end_points
      net = repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      if add_and_check_is_final('conv3', net):
        return net, end_points
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      if add_and_check_is_final('pool3', net):
        return net, end_points
      net = repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      if add_and_check_is_final('conv4', net):
        return net, end_points
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      if add_and_check_is_final('pool4', net):
        return net, end_points
      net = repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      if add_and_check_is_final('conv5', net):
        return net, end_points
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      if add_and_check_is_final('pool5', net):
        return net, end_points
      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
      if add_and_check_is_final('fc6', net):
        return net, end_points
      net = slim.dropout(
          net, dropout_keep_prob, is_training=is_training, scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      if add_and_check_is_final('fc7', net):
        return net, end_points
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        if add_and_check_is_final('global_pool', net):
          return net, end_points
      net = slim.dropout(
          net, dropout_keep_prob, is_training=is_training, scope='dropout7')
      net = slim.conv2d(
          net, num_classes, [1, 1], activation_fn=None, scope='fc8')
      if spatial_squeeze:
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
      end_points[sc.name + '/predictions'] = predictions_fn(net)
      if add_and_check_is_final('fc8', net):
        return net, end_points

    raise ValueError('final_endpoint (%s) not recognized', final_endpoint)


vgg_16.default_image_size = 224
