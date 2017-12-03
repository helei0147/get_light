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

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import sys
import tarfile
import numpy as np
from six.moves import urllib
import tensorflow as tf
from tensorflow.python import debug as tfdbg
import gl_input

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--batch_size', type=int, default=1,
                    help='Number of images to process in a batch.')

parser.add_argument('--data_dir', type=str, default='data_small',
                    help='Path to the CIFAR-10 data directory.')

parser.add_argument('--use_fp16', type=bool, default=False,
                    help='Train the model using fp16.')

FLAGS = parser.parse_args()

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = gl_input.IMAGE_SIZE
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = gl_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = gl_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.6  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.5       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

LIGHT_NUMBER = 5

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  images, labels = gl_input.inputs(eval_data=eval_data,
                                        data_dir='continuous_data/',
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

def check_input(images, lights):
  return images[0,:,:]

def inference(images):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 1, 16],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv3d(images, kernel, [1, 1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)
    tf.Print(tf.shape(conv1), [tf.shape(conv1)])
  # pool1
  pool1 = tf.nn.max_pool3d(conv1, ksize=[1, 2, 2, 1, 1], strides=[1, 2, 2, 1, 1],
                         padding='SAME', name='pool1')
  # norm1
  #norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 16, 32],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv3d(pool1, kernel, [1, 1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

  # norm2
  #norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
  #                  name='norm2')
  # pool2
  pool2 = tf.nn.max_pool3d(conv2, ksize=[1, 2, 2, 1, 1],
                         strides=[1, 2, 2, 1, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, LIGHT_NUMBER*3],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [LIGHT_NUMBER*3],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    #softmax_linear = tf.Print(softmax_linear,[softmax_linear],"\n\n\nsoftmax_linear: ")

    softmax_linear = regularize_normals(softmax_linear)
    #softmax_linear = tf.Print(softmax_linear,[softmax_linear],"\n\n\nsoftmax_linear: ")

    _activation_summary(softmax_linear)

  return softmax_linear

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)
  # by replacing all instances of tf.get_variable() with tf.Variable().
  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op

def loss(est_normals, gts):
  """Calculates the loss from the logits and the labels.

    """
  #tf.assign(est_normals, regularize_normals(est_normals))
  est_normals = regularize_normals(est_normals)
  assert_op = tf.Assert(tf.count_nonzero(gts>10)==0,[gts])
  assert_op.mark_used()
  gts = tf.Print(gts, [gts], 'ground truth: ')
  est_normals = tf.Print(est_normals, [est_normals], 'estimated: ')

  est_colle = tf.split(est_normals, LIGHT_NUMBER, axis=1)
  gts_colle = tf.split(gts, LIGHT_NUMBER, axis=1)
  loss = 0
  for i in range(LIGHT_NUMBER):
    error = tf.multiply(est_colle[i], gts_colle[i])
    cos_error = tf.reduce_sum(error, 1)
    rad_error = tf.acos(cos_error)
    deg_error = rad_error/3.1415926*180
    loss = loss + tf.reduce_sum(deg_error)
  #loss = tf.Print(loss, [loss], '\nloss:')
  return loss

def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: observation tensor, float - [batch_size, 3].
      labels: normal tensor, float - [batch_size, 3]

    Returns:
      total error in degree for this batch
    """
    # regularize estimated normal(logits)
    labels = tf.Print(labels, [labels], 'original light directions:')
    est = regularize_normals(logits)
    est = tf.Print(est, [est], 'estimated: ')
    est_colle = tf.split(est, LIGHT_NUMBER, axis=1)
    gts_colle = tf.split(gts, LIGHT_NUMBER, axis=1)
    for i in range(LIGHT_NUMBER):
      error = tf.multiply(logits, labels)
      cos_error = tf.reduce_sum(error, 1)
      rad_error = tf.acos(cos_error)
      deg_error = rad_error/3.1415926*180
      if i==0:
        total_error = deg_error
      else:
        total_error = tf.add(total_error, deg_error)
    tf.Print(total_error,[total_error], 'total_error')
    # Return the number of true entries.
    return total_error


def regularize_normals(logits):

  normals = tf.split(logits, LIGHT_NUMBER, axis=1)
  real_normals = []
  for normal in normals:
    #normal = tf.Print(normal, [normal], '\nunregularized: ')
    normal = regularize_normals_sub(normal)
    #normal = tf.Print(normal, [normal], '\nregularized normal:')
    real_normals.append(normal)
  logits = tf.concat(real_normals, 1)
  return logits

def regularize_normals_sub(logits):
  dims = tf.split(logits,3, axis=1)
  weight = tf.abs(dims[0])
  weight = tf.concat([weight, weight, weight], axis=1)
  logits = tf.divide(logits, weight)
  #logits = tf.Print(logits, [logits], 'normal:')
  pow_para = tf.zeros(tf.shape(logits))+2
  squared = tf.pow(logits,pow_para)
  sqr_sum = tf.reduce_sum(squared, 1)
  pow_para = tf.zeros(tf.shape(sqr_sum))+0.5
  normal_lengths = tf.pow(sqr_sum,pow_para)
  #normal_lengths = tf.Print(normal_lengths, [normal_lengths], 'normal_len:')
  normal_lengths = tf.expand_dims(normal_lengths,1)
  weight = tf.concat([normal_lengths, normal_lengths, normal_lengths], axis=1)
  #weight = tf.Print(weight, [weight], '\n weight:')
  regulared = tf.divide(logits,weight)
  return regulared
