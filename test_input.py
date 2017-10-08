import os,sys
import tensorflow as tf
import numpy as np
import gl
import gl_input

def test():
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            images, lights = gl.inputs(False)
        p = tf.add_n(images)
        check_op = tf.add_check_numerics_ops()
    with tf.Session() as sess:
        sess.run([p, check_op])
        print(p)


if __name__ == '__main__':
    test()
