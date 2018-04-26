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

"""Evaluation for CIFAR-10.

Accuracy:
gl_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by gl_eval.py.

Speed:
On a single Tesla K40, gl_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import gl

parser = gl.parser

parser.add_argument('--eval_dir', type=str, default='/tmp/gl_eval',
                    help='Directory where to write event logs.')

parser.add_argument('--test_eval_data', type=str, default='test',
                    help='Either `test` or `train_eval`.')

parser.add_argument('--checkpoint_dir', type=str, default='data_tiny',
                    help='Directory where to read model checkpoints.')

parser.add_argument('--eval_interval_secs', type=int, default=60*5,
                    help='How often to run the eval.')

parser.add_argument('--num_examples', type=int, default=1656,
                    help='Number of examples to run.')

parser.add_argument('--run_once', type=bool, default=True,
                    help='Whether to run eval only once.')


def eval_once(saver, summary_writer, result, summary_op, labels, logits):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  prediction_buffer = []
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    print('ckpt', ckpt)
    print('ckpt.model_checkpoint_path', ckpt.model_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint

      # print('\n\nloaded\n\n')
      # saver.restore(sess, ckpt.model_checkpoint_path)
      # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/gl_train/model.ckpt-0,
      # extract global_step from it.


      saver.restore(sess, 'data_tiny/model.ckpt-11253')
      global_step = 11253
    else:
      print('No checkpoint file found')
      return
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord = coord, daemon = True, start = True))
    num_iter = int(math.ceil(FLAGS.num_examples/FLAGS.batch_size))
    total_error = 0
    total_sample_count = num_iter*FLAGS.batch_size
    step = 0
    labels_buffer = []
    prediction_buffer = []
    est_buffer = []
    while step<num_iter and not coord.should_stop():
      prediction = sess.run([result])
      ori_labels, estimated = sess.run([labels, logits])
      # print(prediction)
      ori_labels = np.array(ori_labels)
      estimated = np.array(estimated)
      estimated = np.transpose(estimated, [1,0,2]) # estimated has shape of [batch_size, 4, 3]
      prediction_buffer.append(prediction[0])
      labels_buffer.append(ori_labels)
      est_buffer.append(estimated)
      total_error += np.sum(prediction[0])
      step+=1
    precision = total_error/total_sample_count
    print('precision: %.5f'%(precision))
    summary = tf.Summary()
    summary.ParseFromString(sess.run(summary_op))
    summary.value.add(tag = 'Precision @ 1', simple_value = precision)
    summary_writer.add_summary(summary, global_step)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs = 10)
  prediction_buffer = np.array(prediction_buffer)
  ori_labels = np.array(labels_buffer)
  estimated = np.array(est_buffer)
  return prediction_buffer, ori_labels, estimated


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  fuck_labels = []
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    p_buffer = []
    gt_buffer = []
    est_buffer = []
    eval_data = FLAGS.test_eval_data == 'test'
    images, ratioImages, labels = gl.inputs(eval_data)
    fuck_labels.append(labels)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = gl.inference(images, ratioImages)
    result = gl.evaluation(logits, labels)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(gl.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
    p_cnt = 0
    while True:
      print('\n\neval_once\n\n')
      prediction_buffer, ori_labels, estimated = eval_once(saver, summary_writer, result, summary_op, labels, logits)
      print('---------------------')
      print(ori_labels.shape, estimated.shape)
      print('---------------------')
      p_buffer.append(prediction_buffer)
      gt_buffer.append(ori_labels)
      est_buffer.append(estimated)
      # np.save('predicted/p%d.npy'%(p_cnt), prediction_buffer)
      p_cnt = p_cnt+1
      if FLAGS.run_once:
        break

    np.save('eval_playground/p.npy', np.array(p_buffer)[0])
    np.save('eval_playground/gt.npy', np.array(gt_buffer)[0])
    np.save('eval_playground/est.npy', np.array(est_buffer)[0])

def main(argv=None):  # pylint: disable=unused-argument
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  print('\n\nevaluate\n\n')
  evaluate()


if __name__ == '__main__':
  FLAGS = parser.parse_args()
  tf.app.run()
