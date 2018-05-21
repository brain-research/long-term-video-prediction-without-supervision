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

"""Code for training the hierarchical video prediction model."""

import sys
import time
import prediction_input
import prediction_model
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import debug as tf_debug
from tensorflow.python.platform import app

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_mode', 'e2e', 'Mode to run in. Possible values:'
    "'individual', 'epva', 'epva_gan', 'e2epose_oneop', 'e2epose_sepop', 'e2e'")

flags.DEFINE_integer('pose_dim', 5, 'Dimension of the end effector pose.')
flags.DEFINE_integer('joint_pos_dim', 7, 'Dimension of the joint positions.')

flags.DEFINE_bool('prefetch_enabled', True,
                  'Boolean to enable/disable prefetching')

flags.DEFINE_integer('prefetch_dataset_buffer_size', 256 * 1024 * 1024,
                     'Number of bytes in read buffer. 0 means no buffering.')

flags.DEFINE_integer(
    'cycle_length', 64,
    'Number of elements from dataset to process concurrently '
    '(by interleaver)')

flags.DEFINE_integer(
    'block_length', None,
    'Number of consecutive elements to produce from each input element '
    'before cycling to another input element (by interleaver). '
    'If set to None, block_length defaults to batch_size')

flags.DEFINE_integer('num_parallel_calls', 128,
                     'Number of elements to process in parallel (by mapper)')

flags.DEFINE_integer(
    'initial_shuffle_buffer_size', 1024,
    'Number of elements from dataset that shuffler will sample from. '
    'This shuffling is done before any other operations. '
    'Set to 0 to disable')

flags.DEFINE_integer(
    'followup_shuffle_buffer_size', 128,
    'Number of elements from dataset that shuffler will sample from. '
    'This shuffling is done after prefetching is done. '
    'Set to 0 to disable')

flags.DEFINE_float('enc_keep_prob', 1.0, 'Dropout keep prob for the encoder.')
flags.DEFINE_float('van_keep_prob', 1.0, 'Dropout keep prob for the VAN')
flags.DEFINE_float('enc_noise_stddev', 0, 'Noise between the encoder and VAN')
flags.DEFINE_bool('is_training', False, 'Passed to the VGG encoder')
flags.DEFINE_bool(
    'enc_pred_use_l1_loss', False, 'True to use l1 loss between'
    ' the encoder and predictor instead of l2')

flags.DEFINE_bool(
    'color_data_augment', False, 'Set to true to augment the data'
    ' by randomly changing the hue.')
flags.DEFINE_bool('encoder_grey_in', False, 'True to convert the encoder input'
                  ' to grey scale.')

flags.DEFINE_integer('enc_size', 64, 'The size of the higher level structure.')
flags.DEFINE_float('pred_noise_std', 0.0,
                   'The noise to be fed as additional input to the predictor.')
flags.DEFINE_integer(
    'discrim_steps_per_pred', 5, 'Number of times to train the'
    ' discrim for each train of the predictor.')
flags.DEFINE_bool('use_wgan', True, 'True: Wgan, False: Regular gan')
flags.DEFINE_integer(
    'discrim_context', 1, 'The number of context frames to'
    ' feed into the discrim.')

flags.DEFINE_integer('sequence_length', 10,
                     'sequence length, including context frames.')
flags.DEFINE_integer('skip_num', 1,
                     'Number of frames to skip when reading input')
flags.DEFINE_string(
    'dataset_type', 'human',
    'Controls how data is read in the input pipeline. Possible values:'
    "'robot', 'human'")

flags.DEFINE_string('data_dir', 'gs://unsupervised-hierarch-video/data',
                    'directory containing data.')
flags.DEFINE_string('model_dir', '', 'directory for model checkpoints.')
flags.DEFINE_string('event_log_dir', '', 'directory for writing summary.')
flags.DEFINE_integer('train_steps', 4800000,
                     'Number of steps use for training.')
flags.DEFINE_integer('iterations', 100,
                     'Number of iterations per TPU training loop.')
flags.DEFINE_integer('num_shards', 8, 'Number of shards (TPU chips).')

flags.DEFINE_integer('context_frames', 2, '# of frames before predictions.')

flags.DEFINE_string('data_pattern', '*train*', '')

flags.DEFINE_integer('batch_size', 8,
                     'Global batch size on TPU. Per worker batch size on GPU')

flags.DEFINE_bool('imgnet_pretrain', False,
                  'Whether to pretrain the encoder on imagenet.')
flags.DEFINE_string(
    'epv_pretrain_ckpt',
    'gs://unsupervised-hierarch-video/pretrained_models/epev_human/',
    'The checkpoint to start training from.')

flags.DEFINE_boolean(
    'per_host_input_for_training', True,
    'If true, input_fn is invoked per host rather than per shard.')

flags.DEFINE_float('enc_learning_rate', 1e-5,
                   'Used when the encoder is trained separately.')
flags.DEFINE_float('pred_learning_rate', 3e-4,
                   'Used when the predictor is trained separately.')
flags.DEFINE_float('van_learning_rate', 3e-5,
                   'Used when the VAN is trained separately.')
flags.DEFINE_float('discrim_learning_rate', 1e-2,
                   'Used for the discriminator in epva_gan mode.')
flags.DEFINE_float('all_learning_rate', 1e-5,
                   'Used when multiple parts are trained together.')

flags.DEFINE_float('enc_pred_loss_scale', 1e-2,
                   'The scale of the encoder and predictor loss.')
flags.DEFINE_float('lstm_state_noise_stddev', 0, 'Noise to add to the lstm'
                   ' states in between predictions.')

flags.DEFINE_float(
    'enc_pred_loss_scale_delay', 0,
    'Number of steps for the scale to reach half of its maximum.')
flags.DEFINE_boolean(
    'enc_pred_use_l2norm', False,
    'Use the L2 norm of the encoder and predictor in EPEV mode.')

flags.DEFINE_float('pose_weight', 1,
                   'The weight of the pose loss in the e2e with pose method.')

flags.DEFINE_float('van_r_weight', 0.01,
                   'The weight of the VAN regularization loss.')

flags.DEFINE_float('clip_gradient_norm', 0, '')

flags.DEFINE_bool('use_tpu', False, 'Use TPUs rather than GPU')
flags.DEFINE_bool('use_estimator', False,
                  'True to use tf.estimator. False for slim.')
flags.DEFINE_string('run_mode', 'train',
                    "Mode to run in. Possbile values: 'train', 'eval'")
flags.DEFINE_integer('ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then '
                     'the parameters are handled locally by the worker.')

flags.DEFINE_integer('save_summary_steps', 100,
                     'The frequency with which summaries are saved')
flags.DEFINE_integer('save_checkpoints_secs', 60,
                     'The frequency with which the model is saved, in seconds.')

flags.DEFINE_integer('task', 0, 'Task id of the replica running the training.')
flags.DEFINE_string('master', '', 'BNS name of the TensorFlow master to use.')
flags.DEFINE_integer('startup_delay_secs', 15,
                     'Number of training steps between replicas startup.')

flags.DEFINE_bool('use_image_summary', True,
                  'Whether or not to add the image summary to the graph.')
flags.DEFINE_bool('debug', False, 'Whether to use tf dbg.')
flags.DEFINE_bool('use_legacy_vars', False,
                  'Use outdated tf.Variable instead of tf.get_variable.')


def _get_init_fn():
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """

  if FLAGS.epv_pretrain_ckpt:
    enc_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='timestep/encoder')
    pred_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='timestep/predict')
    van_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='timestep/van')
    all_vars = enc_vars + van_vars + pred_vars

    assignment_map = {}
    for var in all_vars:
      if ('Variable' not in var.op.name) and (
          'back_connect_init' not in var.op.name) and (
              'noise_dense' not in var.op.name):
        assignment_map[var.op.name] = var.op.name

    print 'Fine-tuning from %s' % FLAGS.epv_pretrain_ckpt
    sys.stdout.flush()

    return tf.train.init_from_checkpoint(FLAGS.epv_pretrain_ckpt,
                                         assignment_map)

  elif FLAGS.imgnet_pretrain:
    vgg_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='timestep/encoder/vgg_16')

    assignment_map = {}
    for var in vgg_vars:
      if not var.op.name.startswith('timestep/encoder/vgg_16/fc8'):
        assignment_map[var.op.name[len('timestep/encoder/'):]] = var.op.name

    checkpoint_path = 'gs://unsupervised-hierarch-video/pretrained_models/vgg_16.ckpt'

    print 'Fine-tuning from %s' % checkpoint_path
    sys.stdout.flush()

    return tf.train.init_from_checkpoint(checkpoint_path, assignment_map)


def tf_dbg_sess_wrapper(sess):
  if FLAGS.debug:
    print 'DEBUG'
    sess = tf_debug.LocalCLIDebugWrapperSession(
        sess, thread_name_filter='MainThread$')
    sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)
  return sess


def main(unused_argv):
  if FLAGS.use_tpu:
    run_config = tf.contrib.tpu.RunConfig(
        master=FLAGS.master,
        evaluation_master=FLAGS.master,
        model_dir=FLAGS.model_dir,
        save_checkpoints_secs=FLAGS.save_checkpoints_secs,
        save_summary_steps=FLAGS.save_summary_steps,
        session_config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False),
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations,
            num_shards=FLAGS.num_shards,
            per_host_input_for_training=FLAGS.per_host_input_for_training))

    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=prediction_model.make_model_fn(FLAGS),
        use_tpu=FLAGS.use_tpu,
        config=run_config,
        train_batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.batch_size,
    )
  else:
    run_config = tf.contrib.learn.RunConfig(
        master=FLAGS.master,
        evaluation_master=FLAGS.master,
        model_dir=FLAGS.model_dir,
        save_checkpoints_secs=FLAGS.save_checkpoints_secs,
        save_summary_steps=FLAGS.save_summary_steps,
    )

    estimator = tf.estimator.Estimator(
        model_fn=prediction_model.make_model_fn(FLAGS),
        config=run_config,
    )

  startup_delay_secs = FLAGS.task * FLAGS.startup_delay_secs

  print('delay for:', startup_delay_secs)
  sys.stdout.flush()

  if FLAGS.run_mode == 'train':
    time.sleep(startup_delay_secs)

    if FLAGS.use_estimator or FLAGS.use_tpu:
      print 'using estimator'
      if FLAGS.imgnet_pretrain:
        raise NotImplementedError

      # TODO(wichersn) figure out why estimator doesn't get a good of a loss.
      estimator.train(
          input_fn=prediction_input.get_input_fn(
              FLAGS.data_pattern, FLAGS, FLAGS.batch_size, FLAGS.use_tpu),
          steps=FLAGS.train_steps)
    else:
      print 'using slim'
      # with tf.device(tf.ReplicaDeviceSetter(FLAGS.ps_tasks)):
      features, labels = prediction_input.get_input_fn(
          FLAGS.data_pattern, FLAGS, FLAGS.batch_size, FLAGS.use_tpu)()
      model = prediction_model.make_model_fn(FLAGS)(features, labels, None,
                                                    None)

      saver = tf.train.Saver()

      if FLAGS.task == 0:
        # Only log summaries if it's the chief.
        writer = tf.summary.FileWriter(FLAGS.event_log_dir,
                                       tf.get_default_graph())
      else:
        writer = None

    slim.learning.train(
        model.train_op,
        logdir=FLAGS.event_log_dir,
        saver=saver,
        init_fn=_get_init_fn(),
        save_summaries_secs=FLAGS.save_checkpoints_secs / 2,
        save_interval_secs=FLAGS.save_checkpoints_secs,
        summary_writer=writer,
        number_of_steps=FLAGS.train_steps,
        session_wrapper=tf_dbg_sess_wrapper)

  if FLAGS.run_mode == 'eval':
    features, labels = prediction_input.get_input_fn(
        FLAGS.data_pattern, FLAGS, FLAGS.batch_size, FLAGS.use_tpu)()
    prediction_model.make_model_fn(FLAGS)(features, labels, None, None)

    slim.evaluation.evaluation_loop(
        FLAGS.master,
        FLAGS.model_dir,
        logdir=FLAGS.event_log_dir,
        num_evals=1,
        eval_op=tf.summary.merge_all(),
        eval_interval_secs=FLAGS.save_checkpoints_secs)


if __name__ == '__main__':
  app.run()
