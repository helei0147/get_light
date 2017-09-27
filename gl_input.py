# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

IMAGE_SIZE = 32

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])


def inputs(eval_data, data_dir, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    light_directions: 2D tensor of [batch_size, 3] size.
  """

  # TODO:  need to add train/eval dir difference
  if not eval_data:
    pass
  else:
    pass
  img_folder = data_dir
  light_directions = np.load('/tmp/light_directions.npy')
  npy_buffer = []
  label_buffer = []
  light_direction_buffer = []
  for i in range(200):
      img_name = '%s%d.npy' % (img_folder, i)
      img = np.load(img_name)
      total_channel_num = img.shape[0]*img.shape[3]
      light = light_direction[i, :]
      temp_light_buffer = np.ndarray([total_channel_num, 3], dtype = np.float32) # including the related light directions for all channel images of this light
      temp_light_buffer[:, 0] = light[0]
      temp_light_buffer[:, 1] = light[1]
      temp_light_buffer[:, 2] = light[2]
      npy_buffer.append(img)
      light_direction_buffer.append(temp_light_buffer)

  images = np.concatenate(npy_buffer)
  t = np.transpose(images, axes = [0, 3, 1, 2])
  m = np.concatenate(t, axis = 0)
  images = np.transpose(m, axes = [1, 2, 0])
  reshaped_image = tf.convert_to_tensor(images)
  light_directions = np.concatenate(light_direction_buffer)
  light_directions = tf.convert_to_tensor(light_directions)


  height = IMAGE_SIZE
  width = IMAGE_SIZE
  float_image = reshaped_image


  # Set the shapes of tensors.
  float_image.set_shape([height, width, 3])
  light_directions.set_shape([3])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)
