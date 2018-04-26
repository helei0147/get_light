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
parser.add_argument('--batch_size', type=int, default=64,
                    help='Number of images to process in a batch.')

parser.add_argument('--data_dir', type=str, default='---',
                    help='Path to the CIFAR-10 data directory.')

parser.add_argument('--use_fp16', type=bool, default=False,
                    help='Train the model using fp16.')
parser.add_argument('--eval_data', type=str, default='train',
                    help='if train, use train.use test otherwise')
FLAGS = parser.parse_args()

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = gl_input.IMAGE_SIZE
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = gl_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = gl_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 25     # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.5  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

LIGHT_NUMBER = 4

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
  images, ratioImage, labels = gl_input.inputs(eval_data=eval_data,
                                        data_dir='continuous_data/',
                                        batch_size=FLAGS.batch_size,
                                        if_shuffle=True)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    ratioImage = tf.cast(ratioImage, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, ratioImage, labels

def check_input(images, lights):
  return images[0,:,:]

def batch_norm(inputs_, phase_train=True, decay=0.9, eps=1e-5):
    """Batch Normalization

       Args:
           inputs_: input data(Batch size) from last layer
           phase_train: when you test, please set phase_train "None"
       Returns:
           output for next layer
    """
    gamma = tf.get_variable("gamma", shape=inputs_.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
    beta = tf.get_variable("beta", shape=inputs_.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    pop_mean = tf.get_variable("pop_mean", trainable=False, shape=inputs_.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    pop_var = tf.get_variable("pop_var", trainable=False, shape=inputs_.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
    axes = list(range(len(inputs_.get_shape()) - 1))

    if phase_train != None:
        batch_mean, batch_var = tf.nn.moments(inputs_, axes)
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean*(1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs_, batch_mean, batch_var, beta, gamma, eps)
    else:
        return tf.nn.batch_normalization(inputs_, pop_mean, pop_var, beta, gamma, eps)
def cnn_layers(images):
  with tf.variable_scope('cnns', reuse = tf.AUTO_REUSE):
    # with tf.variable_scope('visualization'):
    #   layer1_image1 = images[0:1,:, :, 0:1]
    #   layer1_image1 = tf.transpose(layer1_image1,perm=[3,1,2,0])
    #   tf.summary.image("filtered_images_layer1",layer1_image1[..., 0::3], max_outputs=2)
    conv1 = tf.layers.conv2d(
        inputs = images,
        filters = 64,
        kernel_size = [3,3],
        padding = 'same',
        strides = 2,
        activation = tf.nn.leaky_relu,
        name = 'cnv1'
    )
    _activation_summary(conv1)
    conv2 = tf.layers.conv2d(
      inputs = conv1,
      filters = 128,
      kernel_size = [3,3],
      padding = 'same',
      strides = 2,
      activation = tf.nn.leaky_relu,
      name = 'cnv2'
    )
    _activation_summary(conv2)
    tf.summary.image("conv2_inter", conv2[0:1,:,:,0:1])
    conv3 = tf.layers.conv2d(
      inputs = conv2,
      filters = 256,
      kernel_size = [3,3],
      padding = 'same',
      strides = 2,
      activation = tf.nn.leaky_relu,
      name = 'cnv3'
    )
    conv3_1 = tf.layers.conv2d(
      inputs = conv3,
      filters = 256,
      kernel_size = [3,3],
      padding = 'same',
      strides = 1,
      activation = tf.nn.leaky_relu,
      name = 'cnv3_1'
    )
    _activation_summary(conv3_1)
    tf.summary.image("conv3_inter", conv2[0:1,:,:,0:3])
    conv4 = tf.layers.conv2d(
      inputs = conv3_1,
      filters = 512,
      kernel_size = [3,3],
      padding = 'same',
      strides = 1,
      activation = tf.nn.leaky_relu,
      name = 'cnv4'
    )
    _activation_summary(conv4)
    output = tf.layers.conv2d(
      inputs = conv4,
      filters = 512,
      kernel_size = [3,3],
      padding = 'same',
      strides = 2,
      name = 'output'
    )

    _activation_summary(output)
  return output

def cnn_layers_way2(images):
  with tf.variable_scope('ratio', reuse = tf.AUTO_REUSE):
    # with tf.variable_scope('visualization'):
    #   layer1_image1 = images[0:1,:, :, 0:1]
    #   layer1_image1 = tf.transpose(layer1_image1,perm=[3,1,2,0])
    #   tf.summary.image("filtered_images_layer1",layer1_image1[..., 0::3], max_outputs=2)
    conv1 = tf.layers.conv2d(
        inputs = images,
        filters = 64,
        kernel_size = [3,3],
        padding = 'same',
        strides = 2,
        activation = tf.nn.leaky_relu,
        name = 'cnv1'
    )
    _activation_summary(conv1)
    conv2 = tf.layers.conv2d(
      inputs = conv1,
      filters = 128,
      kernel_size = [3,3],
      padding = 'same',
      strides = 2,
      activation = tf.nn.leaky_relu,
      name = 'cnv2'
    )
    _activation_summary(conv2)
    tf.summary.image("conv2_inter", conv2[0:1,:,:,0:1])
    conv3 = tf.layers.conv2d(
      inputs = conv2,
      filters = 256,
      kernel_size = [3,3],
      padding = 'same',
      strides = 2,
      activation = tf.nn.leaky_relu,
      name = 'cnv3'
    )
    conv3_1 = tf.layers.conv2d(
      inputs = conv3,
      filters = 256,
      kernel_size = [3,3],
      padding = 'same',
      strides = 1,
      activation = tf.nn.leaky_relu,
      name = 'cnv3_1'
    )
    _activation_summary(conv3_1)
    tf.summary.image("conv3_inter", conv2[0:1,:,:,0:3])
    conv4 = tf.layers.conv2d(
      inputs = conv3_1,
      filters = 512,
      kernel_size = [3,3],
      padding = 'same',
      strides = 1,
      activation = tf.nn.leaky_relu,
      name = 'cnv4'
    )
    _activation_summary(conv4)
    output = tf.layers.conv2d(
      inputs = conv4,
      filters = 512,
      kernel_size = [3,3],
      padding = 'same',
      strides = 2,
      name = 'output'
    )

    _activation_summary(output)
  return output

def build_rcnn_graph(stacked_images, ratioImages):
  NUM_HIDDEN = 1000 #hidden units in lstm
  MAX_STEPSIZE = 4

  cell = tf.contrib.rnn.LSTMCell(NUM_HIDDEN, state_is_tuple=True)
  if FLAGS.eval_data!='test':
    cell = tf.contrib.rnn.DropoutWrapper(cell = cell, output_keep_prob=0.8)

  cell1 = tf.contrib.rnn.LSTMCell(NUM_HIDDEN, state_is_tuple=True)
  if FLAGS.eval_data!='test':
    cell1 = tf.contrib.rnn.DropoutWrapper(cell = cell1, output_keep_prob=0.8)
  # stacking rnn cells
  stack = tf.contrib.rnn.MultiRNNCell([cell, cell1], state_is_tuple=True)

  temp_inputs = []
  reuse=None
  #_activation_summary(stacked_images.shape)
  for i in range(stacked_images.shape[1]):
    conv4 = cnn_layers(stacked_images[:,i, ...])
    I_conv4 = cnn_layers_way2(ratioImages[:,i, ...])
    concatenated_conv = tf.concat([conv4, I_conv4], axis=1)
    temp_inputs.append(conv4)

  # for stack_im in stacked_images:
  #   temp_inputs.append(cnn_layers(stack_im))
  rnn_inputs = [tf.reshape(temp_inputs[i],[FLAGS.batch_size, -1]) for i in range(MAX_STEPSIZE)]
  outputs, states = tf.nn.static_rnn(stack, rnn_inputs, dtype=tf.float32)
  W = tf.get_variable(name='W',
                      shape=[NUM_HIDDEN, 3],
                      dtype=tf.float32,
                      initializer=tf.contrib.layers.xavier_initializer())
  b = tf.get_variable(name='b',
                      shape=[3],
                      dtype=tf.float32,
                      initializer=tf.constant_initializer())
  light_est = [tf.nn.xw_plus_b(output_state, W, b) for output_state in outputs]

  return light_est

def inference(images, ratioImages):
  '''
  images is tensor with shape [32, 32, LIGHT_NUM*3]
  '''
  print(images)
  light_est = build_rcnn_graph(images, ratioImages)

  return light_est

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
    opt = tf.train.AdamOptimizer(lr).minimize(total_loss, global_step=global_step)
    #grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  #apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  apply_gradient_op = opt;

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  # for grad, var in grads:
  #   if grad is not None:
  #     tf.summary.histogram(var.op.name + '/gradients', grad)
  # by replacing all instances of tf.get_variable() with tf.Variable().
  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op

def loss_2(est_normals, gts):
  #est_normals = regularize_normals(est_normals)
  #gts = tf.Print(gts, [gts], 'ground truth: ')
  #est_normals = tf.Print(est_normals, [est_normals], 'estimated: ')
  subs = tf.subtract(est_normals, gts)
  return tf.reduce_sum(tf.multiply(subs, subs))

def loss_depart(est_normals, gts):
  # print('est_normals:', est_normals)
  gts = tf.split(gts, LIGHT_NUMBER, axis=1)
  # print('gts:', gts)
  logits = est_normals
  # print('logits:', logits)
  splited_normals = [regularize(normals) for normals in logits]
  # print(splited_normals)
  loss_result = 0
  for est,gt in zip(splited_normals, gts):
    # est = tf.Print(est, [est], 'est:')
    # gt = tf.Print(gt, [gt], 'gt:')
    to_arccos = tf.reduce_sum(tf.multiply(est, gt), axis=1)
    # REASON FOR THE FOLLOWING TWO LINES
    # max = tf.maximum(to_arccos)
    # temp = to_arccos/max
    # if max>1: # have some error in acos
    #     temp<to_arccos==True
    #     to_arccos = temp
    # else:
    #     to_arccos<temp==True
    #     to_arccos = to_arccos
    #
    # temp = to_arccos/tf.reduce_max(to_arccos)
    # scaled = tf.minimum(temp, to_arccos)
    scaled = to_arccos/1.00001
    degs = tf.acos(scaled)/3.1415926*180
    loss_result = loss_result+tf.reduce_sum(degs)
  los_result = tf.Print(loss_result, [loss_result], 'loss_result')
  return loss_result

def regularize(normals):
    length_square = tf.reduce_sum(tf.square(normals), axis = 1)
    length = tf.sqrt(length_square)
    length = tf.expand_dims(length, 1)
    weight = tf.concat([length, length, length], axis=1)
    print(normals, weight)
    assert weight.shape==normals.shape
    regulared = tf.divide(normals, weight)
    return regulared

def loss_e(est_normals, gts):
  return tf.reduce_sum(tf.exp(tf.abs(tf.subtract(est_normals,gts))))

def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: observation tensor, float - [batch_size, 3].
      labels: normal tensor, float - [batch_size, 3]

    Returns:
      total error in degree for this batch
    """
    total_error = loss_depart(logits, labels)
    return total_error
