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

"""Code for building the input for the prediction model."""

import os

import tensorflow as tf

from tensorflow.python.platform import gfile

COLOR_CHAN = 3
IMG_WIDTH = 64
IMG_HEIGHT = 64

IMAGE_FEATURE_NAME = 'images'
JOINT_POSE_FEATURE_NAME = 'joint_poses'
ACTION_FEATURE_NAME = 'actions'


def get_input_fn(pattern, flags, batch_size, is_tpu):
  """Returns the correct input function for TPU or GPU."""

  def input_fn(params=None):
    """Calls the appropriate input_fn and augments the data."""
    del params
    if is_tpu:
      features = get_input_fn_dataset(pattern, flags, batch_size)()[0]
    else:
      features = get_input_fn_queue(pattern, flags, batch_size)()[0]

    if flags.color_data_augment:

      def augment_img(image):
        image = tf.image.random_hue(image, .5)
        return image

      features[IMAGE_FEATURE_NAME] = tf.map_fn(
          augment_img, features[IMAGE_FEATURE_NAME], parallel_iterations=32)

    return features, None

  return input_fn


def get_input_fn_dataset(pattern, flags, batch_size):
  """Returns input function using Dataset for TPU."""
  print 'Using dataset input fn'

  def input_fn(params=None):
    """Input function using Dataset for TPU."""
    del params
    full_pattern = os.path.join(flags.data_dir, pattern)
    dataset = tf.data.Dataset.list_files(full_pattern)

    if flags.initial_shuffle_buffer_size > 0:
      dataset = dataset.shuffle(buffer_size=flags.initial_shuffle_buffer_size)
    dataset = dataset.repeat()

    # use interleave() and prefetch() to read many files concurrently
    def prefetch_map_fn(filename):
      return tf.data.TFRecordDataset(filename).prefetch(batch_size)

    if flags.prefetch_enabled:
      dataset = dataset.interleave(
          prefetch_map_fn,
          cycle_length=flags.cycle_length,
          block_length=batch_size)

    if flags.followup_shuffle_buffer_size > 0:
      dataset = dataset.shuffle(buffer_size=flags.followup_shuffle_buffer_size)

    frame_nums = range(0, flags.sequence_length, flags.skip_num)

    def parser(_, serialized_example):
      """Parses a single example."""
      features = {}

      for i in frame_nums:
        image_name = 'image_' + str(i)
        if flags.dataset_type == 'robot':
          pose_name = 'state_' + str(i)
          action_name = 'action_' + str(i)
          joint_pos_name = 'joint_positions_' + str(i)

          features[pose_name] = tf.FixedLenFeature([flags.pose_dim], tf.float32)
          features[image_name] = tf.FixedLenFeature([1], tf.string)
          features[action_name] = tf.FixedLenFeature([flags.pose_dim],
                                                     tf.float32)
          features[joint_pos_name] = tf.FixedLenFeature([flags.joint_pos_dim],
                                                        tf.float32)
        else:
          features[image_name] = tf.FixedLenFeature([1], tf.string)

      parsed_input = tf.parse_single_example(serialized_example, features)

      for i in frame_nums:
        image_name = 'image_' + str(i)
        pose_name = 'state_' + str(i)
        action_name = 'action_' + str(i)
        joint_pos_name = 'joint_positions_' + str(i)

        # Process image
        image_buffer = tf.reshape(parsed_input[image_name], shape=[])
        image = tf.image.decode_jpeg(image_buffer, channels=COLOR_CHAN)
        image = tf.image.resize_images(
            image, (IMG_HEIGHT, IMG_WIDTH),
            method=tf.image.ResizeMethod.BICUBIC)
        image = tf.cast(tf.expand_dims(image, 0), tf.float32) / 255.0

        if flags.dataset_type == 'robot':
          pose = tf.reshape(parsed_input[pose_name], shape=[flags.pose_dim])
          pose = tf.expand_dims(pose, 0)
          action = tf.reshape(parsed_input[action_name], shape=[flags.pose_dim])
          action = tf.expand_dims(action, 0)
          joint_pos = tf.reshape(
              parsed_input[joint_pos_name], shape=[flags.joint_pos_dim])
          joint_pos = tf.expand_dims(joint_pos, 0)
        else:
          pose = tf.zeros([1, flags.pose_dim])
          action = tf.zeros([1, flags.pose_dim])
          joint_pos = tf.zeros([1, flags.joint_pos_dim])

        if i == 0:
          image_seq = image
          action_seq, pose_seq, joint_pos_seq = action, pose, joint_pos
        else:
          image_seq = tf.concat([image_seq, image], 0)
          action_seq = tf.concat([action_seq, action], 0)
          pose_seq = tf.concat([pose_seq, pose], 0)
          joint_pos_seq = tf.concat([joint_pos_seq, joint_pos], 0)

      return image_seq, action_seq, action_seq, joint_pos_seq

    dataset = dataset.map(
        parser,
        num_parallel_calls=flags.num_parallel_calls).prefetch(batch_size)

    dataset = dataset.batch(batch_size)

    # use prefetch to overlap producer and consumer
    dataset = dataset.prefetch(1)

    images, actions, poses, joint_pos = dataset.make_one_shot_iterator(
    ).get_next()

    images.set_shape([batch_size, len(frame_nums), IMG_HEIGHT, IMG_WIDTH, 3])
    actions.set_shape([batch_size, len(frame_nums), flags.pose_dim])
    poses.set_shape([batch_size, len(frame_nums), flags.pose_dim])
    joint_pos.set_shape([batch_size, len(frame_nums), flags.joint_pos_dim])

    joint_poses = tf.concat([joint_pos, poses], 2)

    output_features = {
        IMAGE_FEATURE_NAME: images,
        JOINT_POSE_FEATURE_NAME: joint_poses,
        ACTION_FEATURE_NAME: actions
    }

    return output_features, None

  return input_fn


def get_input_fn_queue(pattern, flags, batch_size):
  """Returns input function using queues for GPU."""

  def input_fn(params=None):
    """Input function using queues for GPU."""
    del params
    filenames = gfile.Glob(os.path.join(flags.data_dir, pattern))
    if not filenames:
      raise RuntimeError('No data files found.')
    filename_queue = tf.train.string_input_producer(filenames, shuffle=True)
    reader = tf.TFRecordReader()

    _, val = reader.read(filename_queue)
    serialized_input = tf.reshape(val, shape=[1])

    image_seq = None

    for i in range(0, flags.sequence_length, flags.skip_num):
      image_name = 'image_' + str(i)

      if flags.dataset_type == 'robot':
        pose_name = 'state_' + str(i)
        action_name = 'action_' + str(i)
        joint_pos_name = 'joint_positions_' + str(i)
        features = {
            pose_name:
                tf.FixedLenFeature([flags.pose_dim], tf.float32),
            image_name:
                tf.FixedLenFeature([1], tf.string),
            action_name:
                tf.FixedLenFeature([flags.pose_dim], tf.float32),
            joint_pos_name:
                tf.FixedLenFeature([flags.joint_pos_dim], tf.float32)
        }
      else:
        features = {
            image_name: tf.FixedLenFeature([1], tf.string),
        }

      parsed_input = tf.parse_example(serialized_input, features)

      # Process image
      image_buffer = tf.reshape(parsed_input[image_name], shape=[])
      image = tf.image.decode_jpeg(image_buffer, channels=COLOR_CHAN)
      image = tf.image.resize_images(
          image, (IMG_HEIGHT, IMG_WIDTH), method=tf.image.ResizeMethod.BICUBIC)
      image = tf.cast(tf.expand_dims(image, 0), tf.float32) / 255.0

      if flags.dataset_type == 'robot':
        pose = tf.reshape(parsed_input[pose_name], shape=[flags.pose_dim])
        pose = tf.expand_dims(pose, 0)
        action = tf.reshape(parsed_input[action_name], shape=[flags.pose_dim])
        action = tf.expand_dims(action, 0)
        joint_pos = tf.reshape(
            parsed_input[joint_pos_name], shape=[flags.joint_pos_dim])
        joint_pos = tf.expand_dims(joint_pos, 0)
      else:
        pose = tf.zeros([1, flags.pose_dim])
        action = tf.zeros([1, flags.pose_dim])
        joint_pos = tf.zeros([1, flags.joint_pos_dim])

      if i == 0:
        image_seq = image
        action_seq, pose_seq, joint_pos_seq = action, pose, joint_pos
      else:
        image_seq = tf.concat([image_seq, image], 0)
        action_seq = tf.concat([action_seq, action], 0)
        pose_seq = tf.concat([pose_seq, pose], 0)
        joint_pos_seq = tf.concat([joint_pos_seq, joint_pos], 0)

    [images, actions, poses, joint_pos] = tf.train.shuffle_batch(
        [image_seq, action_seq, pose_seq, joint_pos_seq],
        batch_size,
        num_threads=4,
        capacity=200 * batch_size,
        min_after_dequeue=batch_size * 10,
    )

    joint_poses = tf.concat([joint_pos, poses], 2)

    output_features = {
        IMAGE_FEATURE_NAME: images,
        JOINT_POSE_FEATURE_NAME: joint_poses,
        ACTION_FEATURE_NAME: actions
    }

    return output_features, None

  return input_fn
