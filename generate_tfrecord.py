
import tensorflow as tf
import numpy as np
import os,sys

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value = value))

img_folder = '/tmp/light_npy/'
light_directions = np.load('/tmp/light_directions.npy')
npy_buffer = []
label_buffer = []
light_direction_buffer = []
for i in range(200):
  img_name = '%s%d.npy' % (img_folder, i)
  img = np.load(img_name)
  total_channel_num = img.shape[0]*img.shape[3]
  light = light_directions[i, :]
  temp_light_buffer = np.ndarray([total_channel_num, 3], dtype = np.float32)
  # including the related light directions for all channel images of this light
  temp_light_buffer[:, 0] = light[0]
  temp_light_buffer[:, 1] = light[1]
  temp_light_buffer[:, 2] = light[2]
  npy_buffer.append(img)
  light_direction_buffer.append(temp_light_buffer)

images = np.concatenate(npy_buffer)
t = np.transpose(images, axes = [0, 3, 1, 2])
m = np.concatenate(t, axis = 0)
images = np.transpose(m, axes = [1, 2, 0])
light_directions = np.concatenate(light_direction_buffer)
image_count = images.shape[2]
# write tfrecord
print('\n\n\n')
print(image_count)
print('\n\n\n')
for i in range(image_count):
    file_count = i//100000
    if i%100000==0:
        if file_count==0:
            pass
        else:
            writer.close()
        writer = tf.python_io.TFRecordWriter('data/%d.tfrecord'%(file_count))
    image = images[:,:,i]
    image_list = np.reshape(image, [-1])
    light = light_directions[i,...]
    example = tf.train.Example(features = tf.train.Features(feature = {
        'image':_floats_feature(image_list),
        'light':_floats_feature(light)
    }))
    writer.write(example.SerializeToString())

writer.close()
