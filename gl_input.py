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
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2892
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 20


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
  LIGHT_NUM = 4
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
  print(type(label_batch))
  # Display the training images in the visualizer.
  #tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size, 3*LIGHT_NUM])

def inputs(eval_data, data_dir, batch_size, if_shuffle=False):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, ] size.
    light_directions: 2D tensor of [batch_size, 3] size.
  """

  # TODO:  need to add train/eval dir difference
  if not eval_data:
    filenames = ['curve_tf_train/%d.tfrecord'%(i) for i in range(20)]
    #filenames = ['slim_data_cut/%d.tfrecord'%(i) for i in range(4)]
    print(filenames)
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

  else:
    filenames = ['curve_tf_train/39.tfrecord']
    #filenames = ['slim_data_cut/%d.tfrecord'%(i) for i in range(4,5)]
    print(filenames)
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  feature = {'image': tf.FixedLenFeature([], tf.string),
           'light': tf.FixedLenFeature([], tf.string)}

  filename_queue = tf.train.string_input_producer(filenames)

  reader = tf.TFRecordReader()

  _, serialized_example = reader.read(filename_queue)

  features = tf.parse_single_example(serialized_example, features = feature)

  image = tf.decode_raw(features['image'], tf.float32)
  light = tf.decode_raw(features['light'], tf.float64)

  image = tf.reshape(image,[4,32,32,6])
  light = tf.reshape(light, [12])
  # tf.Assert(tf.count_nonzero(light>10)==0,[light])
#  tf.Print(image, [image], 'image:')
  height = IMAGE_SIZE
  width = IMAGE_SIZE


  # Set the shapes of tensors.
  image.set_shape([4, height, width, 6])
  light.set_shape([12])
  light = tf.cast(light, tf.float32)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(image, light,
                                         min_queue_examples, batch_size,
                                         shuffle=if_shuffle)
