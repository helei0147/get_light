import random
import tensorflow as tf
import numpy as np
import os,sys
import time

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value = value))
def _byte_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

img_folder = 'light_npy_cut/'
npy_buffer = []
picked_number = 5
# read in lights
light_buffer = np.load('light_directions.npy')
light_number = light_buffer.shape[0]
# read in npy
for i in range(light_number):
    img_name = '%s%d.npy' % (img_folder, i)
    img = np.load(img_name)
    npy_buffer.append(img)

npy_buffer = np.array(npy_buffer)
pos_num = npy_buffer.shape[1]
height = npy_buffer.shape[2]
width = npy_buffer.shape[3]
index_buffer = np.arange(pos_num)
index_buffer = index_buffer.tolist()
for f in range(5):
    block_buffer = []
    block_light = []
    writer = tf.python_io.TFRecordWriter('slim_data/%d.tfrecord'%(f))
    for i in range(40000):
        material_index = random.randint(0,19)
        pos_index = random.randint(0,pos_num-1)
        random.shuffle(index_buffer)
        picked_index = index_buffer[0:picked_number]
        picked_light = light_buffer[picked_index, :]
        picked_frames = npy_buffer[picked_index, pos_index, :, :, material_index]
        picked_frames = np.reshape(picked_frames, [picked_number, height, width])
        raw_image_block = picked_frames.tostring()
        raw_light_block = picked_light.tostring()
        example = tf.train.Example(features = tf.train.Features(feature = {
            'image':_byte_feature(tf.compat.as_bytes(raw_image_block)),
            'light':_byte_feature(tf.compat.as_bytes(raw_light_block))
        }))
        writer.write(example.SerializeToString())
    writer.close()
