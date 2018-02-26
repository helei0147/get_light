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
path_container = np.load('path_container.npy')
path_num = path_container.shape[0]
for f in range(5):
    block_buffer = []
    block_light = []
    writer = tf.python_io.TFRecordWriter('continuous_data/%d.tfrecord'%(f))
    npy_frame_buffer = []
    npy_light_buffer = []
    for i in range(40000):
        material_index = random.randint(0,19)
        #pos_index = random.randint(0,pos_num-1)
        pos_index = random.randint(0,pos_num-1)

        # path
        path_index = random.randint(0, 20)
        picked_index = path_container[path_index, :]

        picked_light = light_buffer[picked_index, :]
        picked_frames = npy_buffer[picked_index, pos_index, :, :, material_index]
        picked_frames = np.reshape(picked_frames, [picked_number, height, width])

        canvas = np.zeros([height, width * picked_number])
        for f_ind in range(picked_number):
            canvas[:, f_ind*width:(f_ind+1)*width]=picked_frames[f_ind, :, :]
        pic_name = 'pics/pos_%d-mat_%d-path_%d.png'%(pos_index, material_index, path_index)
        imsave(pic_name, canvas)

        # append to save npy
        npy_frame_buffer.append(picked_frames)
        npy_light_buffer.append(picked_light)

        # convert to tfrecord
        raw_image_block = picked_frames.tostring()
        raw_light_block = picked_light.tostring()
        example = tf.train.Example(features = tf.train.Features(feature = {
            'image':_byte_feature(tf.compat.as_bytes(raw_image_block)),
            'light':_byte_feature(tf.compat.as_bytes(raw_light_block))
        }))
        writer.write(example.SerializeToString())
    writer.close()
    np.save('npy_frames/%d.npy'%(f), np.array(npy_frame_buffer))
    np.save('npy_lights/%d.npy'%(f), np.array(npy_light_buffer))
