# encode:utf-8
import random
import tensorflow as tf
import numpy as np
import os,sys
import time

from scipy.misc import imsave

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value = value))
def _byte_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

for f in range(5):
    blocks = np.load('npy_frames/new_%d.npy'%(f))
    lights = np.load('npy_lights/%d.npy'%(f))
    block_num, frame_num, height, width = blocks.shape
    writer = tf.python_io.TFRecordWriter('new_con_data/%d.tfrecord'%(f))
    for i in range(block_num):
        picked_block = blocks[i,:,:,:]
        picked_light = lights[i,:,:]
        raw_image_block = picked_block.tostring()
        raw_light_block = picked_light.tostring()
        example = tf.train.Example(features = tf.train.Features(feature = {
            'image':_byte_feature(tf.compat.as_bytes(raw_image_block)),
            'light':_byte_feature(tf.compat.as_bytes(raw_light_block))
        }))
        writer.write(example.SerializeToString())
    writer.close()
