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

"""Model architecture for predictive model, including CDNA, DNA, and STP."""

import prediction_input
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.tpu import CrossShardOptimizer
import tf_ops


def van_image_enc_2d(x, first_depth, reuse=False, flags=None):
  """The image encoder for the VAN.

  Similar architecture as Ruben's paper
  (http://proceedings.mlr.press/v70/villegas17a/villegas17a.pdf).

  Args:
    x: The image to encode.
    first_depth: The depth of the first layer. Depth is increased in subsequent
      layers.
    reuse: To reuse in variable scope or not.
    flags: The python flags.

  Returns:
    The encoded image.
  """
  with tf.variable_scope('van_image_enc', reuse=reuse):
    enc_history = [x]

    enc = tf.layers.conv2d(
        x, first_depth, 3, padding='same', activation=tf.nn.relu, strides=1)
    enc = tf.contrib.layers.layer_norm(enc)
    enc = tf.layers.conv2d(
        enc, first_depth, 3, padding='same', activation=tf.nn.relu, strides=1)
    enc = tf.nn.max_pool(enc, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    enc = tf.nn.dropout(enc, flags.van_keep_prob)
    enc = tf.contrib.layers.layer_norm(enc)
    enc_history.append(enc)

    enc = tf.layers.conv2d(
        enc,
        first_depth * 2,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)
    enc = tf.layers.conv2d(
        enc,
        first_depth * 2,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)
    enc = tf.nn.max_pool(enc, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    enc = tf.nn.dropout(enc, flags.van_keep_prob)
    enc = tf.contrib.layers.layer_norm(enc)
    enc_history.append(enc)

    enc = tf.layers.conv2d(
        enc,
        first_depth * 4,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)
    enc = tf.layers.conv2d(
        enc,
        first_depth * 4,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)
    enc = tf.layers.conv2d(
        enc,
        first_depth * 4,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)
    enc = tf.nn.max_pool(enc, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    return enc, enc_history


def van_enc_2d(x, first_depth, reuse=False, flags=None):
  """The higher level structure encoder for the VAN.

  The high level structure is a vector instead of an image.

  Args:
    x: The higher level structure to encode.
    first_depth: The depth of the first layer. Depth is increased in subsequent
      layers.
    reuse: To reuse in variable scope or not.
    flags: The python flags.

  Returns:
    The encoded image.
  """
  del flags
  with tf.variable_scope('van_enc', reuse=reuse):
    enc = tf.nn.relu(x)
    enc = tf.layers.dense(enc, first_depth * 4 * 4, tf.nn.relu)
    enc = tf.contrib.layers.layer_norm(enc)

    enc = tf.reshape(enc, [-1, 4, 4, first_depth])

    enc = tf.layers.conv2d_transpose(
        enc, first_depth, 3, padding='same', activation=tf.nn.relu, strides=1)
    enc = tf.contrib.layers.layer_norm(enc)
    enc = tf.layers.conv2d_transpose(
        enc,
        first_depth * 2,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=2)
    enc = tf.layers.conv2d_transpose(
        enc,
        first_depth * 2,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)
    enc = tf.contrib.layers.layer_norm(enc)
    enc = tf.layers.conv2d_transpose(
        enc,
        first_depth * 4,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)

    return enc


def van_dec_2d(x, skip_connections, output_shape, first_depth, flags=None):
  """The VAN decoder.

  Args:
    x: The analogy information to decode.
    skip_connections: The encoder layers which can be used as skip connections.
    output_shape: The shape of the desired output image.
    first_depth: The depth of the first layer of the van image encoder.
    flags: The python flags.

  Returns:
    The decoded image prediction.
  """
  with tf.variable_scope('van_dec'):
    dec = tf.layers.conv2d_transpose(
        x, first_depth * 4, 3, padding='same', activation=tf.nn.relu, strides=2)
    dec = tf.nn.dropout(dec, flags.van_keep_prob)
    dec = tf.contrib.layers.layer_norm(dec)
    dec = tf.layers.conv2d_transpose(
        dec,
        first_depth * 4,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)
    dec = tf.nn.dropout(dec, flags.van_keep_prob)
    dec = tf.layers.conv2d_transpose(
        dec,
        first_depth * 2,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)
    dec = tf.nn.dropout(dec, flags.van_keep_prob)
    dec = tf.contrib.layers.layer_norm(dec)

    dec = tf.layers.conv2d_transpose(
        dec,
        first_depth * 2,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=2)
    dec = tf.nn.dropout(dec, flags.van_keep_prob)
    dec = tf.layers.conv2d_transpose(
        dec, first_depth, 3, padding='same', activation=tf.nn.relu, strides=1)
    dec = tf.nn.dropout(dec, flags.van_keep_prob)
    dec = tf.contrib.layers.layer_norm(dec)

    dec = tf.layers.conv2d_transpose(
        dec,
        output_shape[3] + 1,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=2)
    dec = tf.nn.dropout(dec, flags.van_keep_prob)

    out_mask = tf.layers.conv2d_transpose(
        dec, output_shape[3] + 1, 3, strides=1, padding='same', activation=None)

    mask = tf.nn.sigmoid(out_mask[:, :, :, 3:4])
    out = out_mask[:, :, :, :3]

    return out * mask + skip_connections[0] * (1 - mask)


def analogy_computation_2d(f_first_enc,
                           f_first_frame,
                           f_current_enc,
                           first_depth,
                           flags=None):
  """Implements the deep analogy computation."""
  with tf.variable_scope('analogy_computation'):
    del flags

    frame_enc_diff = f_first_frame - f_first_enc

    frame_enc_diff_enc = tf.layers.conv2d(
        frame_enc_diff,
        first_depth * 4,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)
    f_current_enc_enc = tf.layers.conv2d(
        f_current_enc,
        first_depth * 4,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)

    analogy = tf.concat([frame_enc_diff_enc, f_current_enc_enc], 3)
    analogy = tf.layers.conv2d(
        analogy,
        first_depth * 4,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)
    analogy = tf.contrib.layers.layer_norm(analogy)
    analogy = tf.layers.conv2d(
        analogy,
        first_depth * 4,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)
    return tf.layers.conv2d(
        analogy,
        first_depth * 4,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)


def van(first_enc, first_frame, current_enc, gt_image, reuse=False, flags=None):
  """Implements a VAN.

  Args:
    first_enc: The first encoding.
    first_frame: The first ground truth frame.
    current_enc: The encoding of the frame to generate.
    gt_image: The ground truth image, only used for regularization.
    reuse: To reuse in variable scope or not.
    flags: The python flags.

  Returns:
    The generated image.
  """
  with tf.variable_scope('van', reuse=reuse):
    output_shape = first_frame.get_shape().as_list()
    output_shape[0] = -1

    first_depth = 64

    f_first_enc = van_enc_2d(first_enc, first_depth, flags=flags)
    f_first_frame, image_enc_history = van_image_enc_2d(
        first_frame, first_depth, flags=flags)
    f_current_enc = van_enc_2d(
        current_enc, first_depth, reuse=True, flags=flags)
    f_gt_image, _ = van_image_enc_2d(gt_image, first_depth, True, flags=flags)

    analogy_t = analogy_computation_2d(
        f_first_enc, f_first_frame, f_current_enc, first_depth, flags=flags)
    enc_img = f_current_enc + analogy_t

    img = van_dec_2d(
        enc_img, image_enc_history, output_shape, first_depth, flags=flags)

    batch_size = tf.to_float(tf.shape(first_enc)[0])
    r_loss = tf.nn.l2_loss(f_gt_image - f_current_enc - analogy_t) / batch_size

    return img, r_loss


def encoder_vgg(x, enc_final_size, reuse=False, scope_prefix='', flags=None):
  """VGG network to use as encoder without the top few layers.

  Can be pretrained.

  Args:
    x: The image to encode. In the range 0 to 1.
    enc_final_size: The desired size of the encoding.
    reuse: To reuse in variable scope or not.
    scope_prefix: The prefix before the scope name.
    flags: The python flags.

  Returns:
    The generated image.
  """
  with tf.variable_scope(scope_prefix + 'encoder', reuse=reuse):

    # Preprocess input
    x *= 256
    x = [x[:, :, :, 0] - 123.68, x[:, :, :, 1] - 116.78, x[:, :, :, 2] - 103.94]
    x = tf.transpose(x, [1, 2, 3, 0])

    with slim.arg_scope(tf_ops.vgg_arg_scope()):
      enc, _ = tf_ops.vgg_16(
          x,
          num_classes=enc_final_size,
          is_training=flags.is_training,
          final_endpoint='pool5')

    enc_shape = enc.get_shape().as_list()
    enc_shape[0] = -1
    enc_size = enc_shape[1] * enc_shape[2] * enc_shape[3]

    enc_flat = tf.reshape(enc, (-1, enc_size))
    enc_flat = tf.nn.dropout(enc_flat, flags.enc_keep_prob)

    enc_flat = tf.layers.dense(
        enc_flat,
        enc_final_size,
        kernel_initializer=tf.truncated_normal_initializer(stddev=1e-4,))

    if flags.enc_pred_use_l2norm:
      enc_flat = tf.nn.l2_normalize(enc_flat, 1)

  return enc_flat


def predictor(enc_flat,
              action,
              lstm_states,
              pred_depth,
              reuse=False,
              scope_prefix='',
              flags=None):
  """LSTM predictor network."""
  with tf.variable_scope(scope_prefix + 'predict', reuse=reuse):
    enc_final_size = enc_flat.get_shape().as_list()[1]
    action_size = action.get_shape().as_list()[1]
    initial_size = (enc_final_size + action_size)

    batch_size = enc_flat.get_shape().as_list()[0]

    init_stddev = 1e-2

    pre_pred = tf.concat([enc_flat, action], 1)
    pre_pred = tf.layers.dense(
        pre_pred,
        initial_size,
        tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=init_stddev))

    if lstm_states[pred_depth - 2] is None:
      if not flags.use_legacy_vars:
        back_connect = [
            tf.get_variable(
                'back_connect_init',
                shape=[initial_size * 2],
                initializer=tf.truncated_normal_initializer(stddev=init_stddev))
        ] * batch_size
      else:
        back_connect = [
            tf.Variable(
                tf.random_normal(shape=[initial_size * 2], stddev=init_stddev))
        ] * batch_size
    else:
      back_connect = lstm_states[pred_depth - 2]

    part_pred, lstm_states[0] = tf_ops.lstm_cell(
        tf.concat([pre_pred, back_connect], 1),
        lstm_states[0],
        initial_size,
        use_peepholes=True,
        initializer=tf.truncated_normal_initializer(stddev=init_stddev),
        num_proj=initial_size)
    part_pred = tf.contrib.layers.layer_norm(part_pred)
    pred = part_pred

    for pred_layer_num in range(1, pred_depth, 2):
      part_pred, lstm_states[pred_layer_num] = tf_ops.lstm_cell(
          pred,
          lstm_states[pred_layer_num],
          initial_size,
          use_peepholes=True,
          initializer=tf.truncated_normal_initializer(stddev=init_stddev),
          num_proj=initial_size)
      pred += part_pred

      part_pred, lstm_states[pred_layer_num + 1] = tf_ops.lstm_cell(
          tf.concat([pred, pre_pred], 1),
          lstm_states[pred_layer_num + 1],
          initial_size,
          use_peepholes=True,
          initializer=tf.truncated_normal_initializer(stddev=init_stddev),
          num_proj=initial_size)
      part_pred = tf.contrib.layers.layer_norm(part_pred)
      pred += part_pred

    pred = tf.layers.dense(
        pred,
        enc_final_size,
        kernel_initializer=tf.truncated_normal_initializer(stddev=init_stddev))

    if flags.enc_pred_use_l2norm:
      pred = tf.nn.l2_normalize(pred, 1)

    return pred


def get_pose_from_enc(enc, pose_size):
  return enc[:, :pose_size]


class ModelOutputs(object):
  """Struct to organize the model outputs."""

  def __init__(self):
    self.enc_out_all = []
    self.pose_from_enc_all = []
    self.pred_out_all = []
    self.pose_from_pred_all = []
    self.pred_on_pose_out_all = []
    self.pose_from_pred_on_pose_all = []
    self.van_out_all = []
    self.van_on_enc_all = []
    self.van_on_pose_all = []


def construct_model(images,
                    actions=None,
                    poses=None,
                    iter_num=-1.0,
                    context_frames=2,
                    flags=None):
  """Constructs the tensorflow graph of the hierarchical model."""
  del iter_num

  pred_depth = 20
  if flags.model_mode == 'epev_gann':
    pred_depth = 2

  model_outputs = ModelOutputs()

  lstm_states = [None] * (pred_depth + 2)

  pose_size = flags.pose_dim + flags.joint_pos_dim

  enc_out = encoder_vgg(
      images[0], flags.enc_size, False, scope_prefix='timestep/', flags=flags)
  enc_out = tf.identity(enc_out, 'enc_out')
  model_outputs.enc_out_all.append(enc_out)
  model_outputs.pose_from_enc_all.append(get_pose_from_enc(enc_out, pose_size))

  num_timesteps = len(actions) - 1
  sum_freq = int(num_timesteps / 4 + 1)

  reuse = False
  for timestep, action in zip(range(len(actions) - 1), actions[:-1]):
    done_warm_start = timestep > context_frames - 1

    with tf.variable_scope('timestep', reuse=reuse):
      if done_warm_start:
        pred_input = model_outputs.pred_out_all[-1]
      else:
        pred_input = model_outputs.enc_out_all[-1]
      pred_out = predictor(
          pred_input, action, lstm_states, pred_depth, False, flags=flags)
      pred_out = tf.identity(pred_out, 'pred_out')
      if timestep % sum_freq == 0 and not flags.use_tpu:
        tf.summary.histogram('pred_out', pred_out)
      model_outputs.pred_out_all.append(pred_out)
      model_outputs.pose_from_pred_all.append(
          get_pose_from_enc(pred_out, pose_size))

      if flags.enc_size == pose_size:
        if done_warm_start:
          pred_on_pose_input = model_outputs.pred_on_pose_out_all[-1]
        else:
          pred_on_pose_input = poses[timestep]
        pred_on_pose_out = predictor(
            pred_on_pose_input,
            action,
            lstm_states,
            pred_depth,
            True,
            flags=flags)
        pred_on_pose_out = tf.identity(pred_on_pose_out, 'pred_on_pose_out')
        model_outputs.pred_on_pose_out_all.append(pred_on_pose_out)
        model_outputs.pose_from_pred_on_pose_all.append(
            get_pose_from_enc(pred_on_pose_out, pose_size))

      van_out, _ = van(
          model_outputs.enc_out_all[0],
          images[0],
          pred_out,
          images[timestep + 1],
          False,
          flags=flags)
      van_out = tf.identity(van_out, 'van_out')
      model_outputs.van_out_all.append(van_out)

      enc_out = encoder_vgg(
          images[timestep + 1], flags.enc_size, True, flags=flags)
      enc_out = tf.identity(enc_out, 'enc_out')
      if timestep % sum_freq == 0 and not flags.use_tpu:
        tf.summary.histogram('enc_out', enc_out)
      model_outputs.enc_out_all.append(enc_out)
      model_outputs.pose_from_enc_all.append(
          get_pose_from_enc(enc_out, pose_size))

      # van_input = tf.cond(tf.greater(iter_num, 10000),
      #                     lambda: images[0], lambda: tf.zeros_like(images[0]))
      van_input = images[0]
      if flags.enc_noise_stddev > 0:
        enc_noise = tf.truncated_normal(
            enc_out.shape, stddev=flags.enc_noise_stddev)
      else:
        enc_noise = tf.zeros_like(enc_out)
      if timestep % sum_freq == 0 and not flags.use_tpu:
        tf.summary.histogram('enc_noise', enc_noise)
      van_on_enc, _ = van(
          model_outputs.enc_out_all[0],
          van_input,
          enc_out + enc_noise,
          images[timestep + 1],
          True,
          flags=flags)
      van_on_enc = tf.identity(van_on_enc, 'van_on_enc')
      model_outputs.van_on_enc_all.append(van_on_enc)

      if flags.enc_size == pose_size:
        van_on_pose, _ = van(
            poses[0],
            images[0],
            poses[timestep + 1],
            images[timestep + 1],
            True,
            flags=flags)
        van_on_pose = tf.identity(van_on_pose, 'van_on_pose')
        model_outputs.van_on_pose_all.append(van_on_pose)

      reuse = True

  return model_outputs


def make_model_fn(flags):
  """Returns model_fn."""

  def model_fn(features, labels, mode, params):
    """Model with losses, compatible with tf.estimator."""
    del labels, params

    images = features[prediction_input.IMAGE_FEATURE_NAME]
    joint_poses = features[prediction_input.JOINT_POSE_FEATURE_NAME]
    actions = features[prediction_input.ACTION_FEATURE_NAME]

    actions = tf.split(
        value=actions, num_or_size_splits=actions.shape[1], axis=1)
    actions = [tf.squeeze(act, 1) for act in actions]
    joint_poses = tf.split(
        value=joint_poses, num_or_size_splits=joint_poses.shape[1], axis=1)
    joint_poses = [tf.squeeze(st, 1) for st in joint_poses]
    images = tf.split(value=images, num_or_size_splits=images.shape[1], axis=1)
    images = [tf.squeeze(img, 1) for img in images]

    model_outputs = construct_model(
        images,
        actions,
        joint_poses,
        iter_num=slim.get_or_create_global_step(),
        context_frames=flags.context_frames,
        flags=flags)

    pose_from_enc_loss = calc_pose_loss(
        model_outputs.pose_from_enc_all,
        joint_poses,
        'pose_from_enc_loss_quant',
        flags=flags)

    pose_from_pred_loss = calc_pose_loss(
        model_outputs.pose_from_pred_all,
        joint_poses[1:],
        'pose_from_pred_loss_quant',
        flags=flags)

    if model_outputs.pose_from_pred_on_pose_all:
      pose_from_pred_on_pose_loss = calc_pose_loss(
          model_outputs.pose_from_pred_on_pose_all,
          joint_poses[1:],
          'pose_from_pred_on_pose_loss_quant',
          flags=flags)

    enc_pred_loss, _ = calc_loss_psnr(
        model_outputs.enc_out_all[1:],
        model_outputs.pred_out_all,
        'enc_pred_loss',
        flags=flags)

    van_loss, _ = calc_loss_psnr(
        model_outputs.van_out_all, images[1:], 'van_loss', flags=flags)
    van_on_enc_loss, _ = calc_loss_psnr(
        model_outputs.van_on_enc_all,
        images[1:],
        'van_on_enc_loss',
        flags=flags)
    if model_outputs.van_on_pose_all:
      van_on_pose_loss, _ = calc_loss_psnr(
          model_outputs.van_on_pose_all,
          images[1:],
          'van_on_pose_loss',
          flags=flags)

    if flags.use_image_summary and (not flags.use_tpu):
      add_video_summaries(images[1:], model_outputs.van_out_all, 'van_out_all')
      add_video_summaries(images[1:], model_outputs.van_on_enc_all,
                          'van_on_enc_all')
      add_video_summaries(images[1:], model_outputs.van_on_pose_all,
                          'van_on_pose_all')

    enc_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='timestep/encoder')
    pred_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='timestep/predict')
    van_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='timestep/van')
    all_vars = enc_vars + pred_vars + van_vars

    global_step = tf.train.get_or_create_global_step()
    increment_global_step = tf.assign(global_step, global_step + 1)

    if flags.model_mode == 'epev':
      enc_pred_loss_scale_delay = max(flags.enc_pred_loss_scale_delay, 1)
      enc_pred_loss_scale = tf.nn.sigmoid(
          (tf.to_float(tf.train.get_or_create_global_step()
                      ) - enc_pred_loss_scale_delay) /
          (enc_pred_loss_scale_delay * .1)) * flags.enc_pred_loss_scale
      tf.summary.scalar('enc_pred_loss_scale', enc_pred_loss_scale)
      epev_loss = enc_pred_loss * enc_pred_loss_scale + van_on_enc_loss
      tf.summary.scalar('epev_loss', epev_loss)

    all_train_op = None
    if flags.is_training:
      enc_optimizer = tf.train.AdamOptimizer(flags.enc_learning_rate)
      pred_optimizer = tf.train.AdamOptimizer(flags.pred_learning_rate)
      van_optimizer = tf.train.AdamOptimizer(flags.van_learning_rate)
      all_optimizer = tf.train.AdamOptimizer(flags.all_learning_rate)

      if flags.use_tpu:
        enc_optimizer = CrossShardOptimizer(enc_optimizer)
        pred_optimizer = CrossShardOptimizer(pred_optimizer)
        van_optimizer = CrossShardOptimizer(van_optimizer)
        all_optimizer = CrossShardOptimizer(all_optimizer)

      if flags.use_legacy_vars:
        fake_step = tf.Variable(0, dtype=tf.int32)
      else:
        fake_step = tf.Variable(0, dtype=tf.float32)

      if flags.model_mode == 'individual':
        enc_train_op = slim.learning.create_train_op(
            pose_from_enc_loss,
            enc_optimizer,
            clip_gradient_norm=flags.clip_gradient_norm,
            variables_to_train=enc_vars,
            global_step=fake_step)
        pred_train_op = slim.learning.create_train_op(
            pose_from_pred_on_pose_loss,
            pred_optimizer,
            clip_gradient_norm=flags.clip_gradient_norm,
            variables_to_train=pred_vars,
            global_step=fake_step)

      if flags.model_mode == 'e2epose_sepop':
        enc_train_op = slim.learning.create_train_op(
            pose_from_enc_loss,
            enc_optimizer,
            clip_gradient_norm=flags.clip_gradient_norm,
            variables_to_train=enc_vars,
            global_step=fake_step)
        pred_train_op = slim.learning.create_train_op(
            pose_from_pred_loss,
            pred_optimizer,
            clip_gradient_norm=flags.clip_gradient_norm,
            variables_to_train=pred_vars,
            global_step=fake_step)

      if flags.model_mode == 'individual':
        van_train_op = slim.learning.create_train_op(
            van_on_pose_loss,
            van_optimizer,
            clip_gradient_norm=flags.clip_gradient_norm,
            variables_to_train=van_vars,
            global_step=fake_step)
        all_train_op = tf.tuple(
            [van_loss],
            control_inputs=[
                enc_train_op, pred_train_op, van_train_op, increment_global_step
            ])[0]

      if flags.model_mode == 'epev':
        all_train_op = slim.learning.create_train_op(
            epev_loss,
            all_optimizer,
            clip_gradient_norm=flags.clip_gradient_norm,
            variables_to_train=all_vars)

      if flags.model_mode == 'e2epose_oneop':
        all_train_op = slim.learning.create_train_op(
            (pose_from_enc_loss + pose_from_pred_loss) * flags.pose_weight +
            van_loss,
            all_optimizer,
            clip_gradient_norm=flags.clip_gradient_norm,
            variables_to_train=all_vars)

      if flags.model_mode == 'e2epose_sepop':
        e2e_train_op = slim.learning.create_train_op(
            van_loss,
            all_optimizer,
            clip_gradient_norm=flags.clip_gradient_norm,
            global_step=fake_step,
            variables_to_train=all_vars)
        all_train_op = tf.tuple(
            [van_loss],
            control_inputs=[
                e2e_train_op, enc_train_op, pred_train_op, increment_global_step
            ])[0]

      if flags.model_mode == 'e2e':
        all_train_op = slim.learning.create_train_op(
            van_loss,
            all_optimizer,
            clip_gradient_norm=flags.clip_gradient_norm,
            variables_to_train=all_vars)

    predictions = None
    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions_lst = {
          'gt_images': images,
          'van_out_all': model_outputs.van_out_all,
          'van_on_enc_all': model_outputs.van_on_enc_all
      }

      predictions = {}
      for key in predictions_lst:
        if predictions_lst[key]:
          predictions[key] = tf.convert_to_tensor(predictions_lst[key])

          predictions[key] = tf.transpose(
              predictions[key],
              tf.concat([[1, 0], tf.range(2, tf.rank(predictions[key]))], 0))

    if flags.use_tpu:
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=van_loss,
          train_op=all_train_op,
          predictions=predictions)
    else:
      return tf.estimator.EstimatorSpec(
          mode=mode,
          loss=van_loss,
          train_op=all_train_op,
          predictions=predictions)

  return model_fn


def peak_signal_to_noise_ratio(true, pred):
  """Image quality metric based on maximal signal power vs. power of the noise.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    peak signal to noise ratio (PSNR)
  """
  return 10.0 * tf.log(1.0 / mean_squared_error(true, pred)) / tf.log(10.0)


def mean_squared_error(true, pred):
  """L2 distance between tensors true and pred.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """
  result = tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))
  return result


def l1_error(true, pred):
  """L1 distance between tensors true and pred."""
  return tf.reduce_sum(tf.abs(true - pred)) / tf.to_float(tf.size(pred))


def calc_loss_psnr(gen_images, images, name, flags=None):
  """Calculates loss and psnr for predictions over multiple timesteps."""
  with tf.name_scope(name):
    loss, error, psnr_all = 0.0, 0.0, 0.0
    for _, x, gx in zip(range(len(gen_images)), images, gen_images):
      recon_cost = mean_squared_error(x, gx)

      error_i = l1_error(x, gx)
      psnr_i = peak_signal_to_noise_ratio(x, gx)
      psnr_all += psnr_i
      error += error_i
      loss += recon_cost

    psnr_all /= tf.to_float(len(gen_images))
    loss /= tf.to_float(len(gen_images))
    error /= tf.to_float(len(gen_images))

    if not flags.use_tpu:
      tf.summary.scalar('psnr_all', psnr_all)
      tf.summary.scalar('loss', loss)

    return loss, psnr_all


def calc_pose_loss(gen_poses, poses, name, flags):
  """Calculates the loss assuming the inputs are the pose of the robot arm."""
  with tf.name_scope(name):

    batch_size = tf.shape(gen_poses[0])[0]
    joint_loss, joint_error, ee_loss, quant_angle_loss = 0.0, 0.0, 0.0, 0.0
    for _, x, gx in zip(range(len(gen_poses)), poses, gen_poses):

      if x.get_shape().as_list() != gx.get_shape().as_list():
        return tf.constant(0, dtype=tf.float32)

      x_joint, x_ee, x_quant_angles = pose_angles_to_quaternion(x, flags)
      gx_joint, gx_ee, gx_quant_angles = pose_angles_to_quaternion(gx, flags)

      joint_loss_i = mean_squared_error(x_joint, gx_joint)
      joint_error_i = l1_error(x_joint, gx_joint)
      ee_loss_i = mean_squared_error(x_ee, gx_ee)
      quant_angle_loss_i = calc_quaternion_loss(gx_quant_angles, x_quant_angles,
                                                {
                                                    'use_logging': True,
                                                    'batch_size': batch_size
                                                })

      joint_loss += joint_loss_i
      joint_error += joint_error_i
      ee_loss += ee_loss_i
      quant_angle_loss += quant_angle_loss_i

    joint_loss /= tf.to_float(len(gen_poses))
    joint_error /= tf.to_float(len(gen_poses))
    ee_loss /= tf.to_float(len(gen_poses))
    quant_angle_loss /= tf.to_float(len(gen_poses))

    if not flags.use_tpu:
      tf.summary.scalar('loss_joint', joint_loss)
      tf.summary.scalar('joint_error', joint_error)
      tf.summary.scalar('loss_ee', ee_loss)
      tf.summary.scalar('loss_quant_angle', quant_angle_loss)

    total = joint_loss + ee_loss + quant_angle_loss
    if not flags.use_tpu:
      tf.summary.scalar('total_loss', total)
    return total


def quaternion_from_euler(ai, aj, ak):
  """Converts in szyx mode."""

  ai = tf.to_float(ai) / 2.0
  aj = tf.to_float(aj) / 2.0
  ak = tf.to_float(ak) / 2.0
  ci = tf.cos(ai)
  si = tf.sin(ai)
  cj = tf.cos(aj)
  sj = tf.sin(aj)
  ck = tf.cos(ak)
  sk = tf.sin(ak)
  cc = ci * ck
  cs = ci * sk
  sc = si * ck
  ss = si * sk

  quaternion = [
      cj * sc - sj * cs, cj * ss + sj * cc, cj * cs - sj * sc, cj * cc + sj * ss
  ]
  return tf.transpose(quaternion)


def pose_angles_to_quaternion(pose, flags):
  """Converts the end effector positions of the pose to quaternions."""
  quant_angles = quaternion_from_euler(
      tf.zeros_like(tf.squeeze(pose[:, -2])), tf.squeeze(pose[:, -2]),
      tf.squeeze(pose[:, -1]))
  return pose[:, :flags.joint_pos_dim], pose[:, flags.joint_pos_dim:
                                             -2], quant_angles


def quaternion_loss_batch(predictions, labels, params):
  """A helper function to compute the error between quaternions.

  Args:
    predictions: A Tensor of size [batch_size, 4].
    labels: A Tensor of size [batch_size, 4].
    params: A dictionary of parameters. Expecting 'use_logging', 'batch_size'.

  Returns:
    A Tensor of size [batch_size], denoting the error between the quaternions.
  """
  use_logging = params['use_logging']
  assertions = []
  if use_logging:
    assertions.append(
        tf.Assert(
            tf.reduce_all(
                tf.less(
                    tf.abs(tf.reduce_sum(tf.square(predictions), [1]) - 1),
                    1e-4)),
            ['The l2 norm of each prediction quaternion vector should be 1.']))
    assertions.append(
        tf.Assert(
            tf.reduce_all(
                tf.less(
                    tf.abs(tf.reduce_sum(tf.square(labels), [1]) - 1), 1e-4)),
            ['The l2 norm of each label quaternion vector should be 1.']))

  with tf.control_dependencies(assertions):
    product = tf.multiply(predictions, labels)
  internal_dot_products = tf.reduce_sum(product, [1])

  return 1 - internal_dot_products


def calc_quaternion_loss(predictions, labels, params):
  """A helper function to compute the mean error between batches of quaternions.

  The caller is expected to add the loss to the graph.

  Args:
    predictions: A Tensor of size [batch_size, 4].
    labels: A Tensor of size [batch_size, 4].
    params: A dictionary of parameters. Expecting 'use_logging', 'batch_size'.

  Returns:
    A Tensor of size 1, denoting the mean error between batches of quaternions.
  """
  cost = quaternion_loss_batch(predictions, labels, params)
  cost = tf.reduce_sum(cost, [0])
  batch_size = tf.to_float(params['batch_size'])
  cost = tf.multiply(cost, 1.0 / batch_size, name='quaternion_loss')
  return cost


def add_video_summaries(images, gen_images, name):
  del images, gen_images, name


#  with tf.device('/cpu:0'):
#    if gen_images:
#      if images is not None:
#        summary_ops.AnimatedWebpSummary(
#            name + '/real',
#            tf.concat(images, 3) * 255,
#            channels=3,
#            min_range=0,
#            max_range=255,
#            max_animations=4,
#            frame_length_ms=100)
#      summary_ops.AnimatedWebpSummary(
#          name + '/gen',
#          tf.concat(gen_images, 3) * 255,
#          channels=3,
#          min_range=0,
#          max_range=255,
#          max_animations=4,
#          frame_length_ms=100)
