# encode:utf-8
import random
import tensorflow as tf
import numpy as np
import os,sys
import time

from scipy.misc import imsave

LIGHT_NUMBER = 5

TEST_FLAG = False;

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value = value))
def _byte_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def count_nan(matrix):
    nan_count = np.count_nonzero(np.isnan(matrix))
    if nan_count>0:
        print('nan occurs, nan count is %d'%(nan_count))
    return nan_count
nan_info_ori = []
nan_info_I = []
if TEST_FLAG:
    npy_file_num = 8
else:
    npy_file_num = 40
for f in range(npy_file_num):
    if TEST_FLAG:
        blocks = np.load('test/boosted/new_%d.npy'%(f))
        lights = np.load('test/lights/%d.npy'%(f))
        writer = tf.python_io.TFRecordWriter('curve_tf_test/%d.tfrecord'%(f))
    else:
        blocks = np.load('train/boosted/new_%d.npy'%(f))
        lights = np.load('train/lights/%d.npy'%(f))
        writer = tf.python_io.TFRecordWriter('curve_tf_train/%d.tfrecord'%(f))

    block_num, frame_num, height, width = blocks.shape
    print(blocks.shape)
    shuffle_index = np.arange(block_num)
    np.random.shuffle(shuffle_index)
    blocks = blocks[shuffle_index,:,:,:]
    lights = np.reshape(lights, [block_num, 6, 3])
    lights = lights[shuffle_index,:,:]
    lights = np.reshape(lights, [-1, 3])
    for i in range(block_num):
        picked_block = blocks[i,:,:,:]
        #print(picked_block.shape)
        # WAY 1
        buf = []
        for j in range(LIGHT_NUMBER-1):
            temp = picked_block[j*3:j*3+6, ...]
            #print(temp.shape)
            assert temp.shape==(6,32,32)
            buf.append(temp)
        buf = np.array(buf)
        buf = np.transpose(buf, [0, 2, 3, 1])# shape is (step, xsize, ysize, channel)
        assert buf.shape==(LIGHT_NUMBER-1, 32, 32, 6)

        # WAY 2
        I_buffer = []
        I_block = picked_block[LIGHT_NUMBER*3:, ...]
        assert I_block.shape[0]==LIGHT_NUMBER*3
        # channels for I, Ix, Iy, I_deltaT, delta(I_deltaT-I)
        block_canvas = np.zeros([5, height, width])
        for j in range(LIGHT_NUMBER-1):
            I_Ix_Iy = I_block[j::LIGHT_NUMBER]
            I_delta = I_block[j+1::LIGHT_NUMBER]
            delta = I_delta[0, ...]-I_Ix_Iy[0, ...]
            block_canvas[0:3, :, :] = I_Ix_Iy
            block_canvas[3, :, :] = I_delta[0, :, :]
            block_canvas[4, :, :] = delta
            I_buffer.append(block_canvas)
        I_buffer = np.array(I_buffer)
        I_buffer = np.transpose(I_buffer, [0, 2, 3, 1])
        assert I_buffer.shape==(LIGHT_NUMBER-1, 32, 32, 5)

        picked_light = lights[i*LIGHT_NUMBER:i*LIGHT_NUMBER+LIGHT_NUMBER-1,:]
        picked_light = np.reshape(picked_light, [-1])
        assert picked_light.shape==(12,)

        nan_count_ori = count_nan(buf)
        nan_count_I = count_nan(I_buffer)
        if nan_count_ori>0 :
            print('nan_ori in file: %d, block_index: %d, count: %d'%(npy_file_num, i, nan_count_ori))
            nan_info_ori.append([npy_file_num, i, nan_count_ori])
        if nan_count_I>0:
            print('nan_I in file: %d, block_index: %d, count: %d'%(npy_file_num, i, nan_count_I))
            nan_info_I.append([npy_file_num, i, nan_count_I])
        raw_image_block = buf.tostring()
        raw_RatioImage_block = I_buffer.tostring()
        raw_light_block = picked_light.tostring()
        example = tf.train.Example(features = tf.train.Features(feature = {
            'image':_byte_feature(tf.compat.as_bytes(raw_image_block)),
            'ratioImage':_byte_feature(tf.compat.as_bytes(raw_RatioImage_block)),
            'light':_byte_feature(tf.compat.as_bytes(raw_light_block))
        }))
        writer.write(example.SerializeToString())
    writer.close()

np.save('nan_info_ori.npy',np.array(nan_info_ori))
np.save('nan_info_I.npy', np.array(nan_info_I))
