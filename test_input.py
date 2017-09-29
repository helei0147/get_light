import os,sys
import tensorflow as tf
import numpy as np

def test():
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            images, labels = gl.inputs(False)
        check_input_op = gl.check_input()
        with tf.Session() as sess:
            sess.run(check_input_op)
