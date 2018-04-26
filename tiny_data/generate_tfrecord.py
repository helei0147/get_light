
import tensorflow as tf
import numpy as np
import os,sys
import time
def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value = value))
def _byte_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
img_folder = 'light_npy/'
light_directions = np.load('light_directions.npy')
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
images = np.transpose(images, axes = [0,3,1,2])
images = np.concatenate(images)
print(images.shape)
images = np.transpose(images, axes = [1, 2, 0])
light_directions = np.concatenate(light_direction_buffer)
image_count = images.shape[2]
# images is a array with shape [32,32,image_count]
# light_directions is an array with shape [image_count, 3]

# shuffle the input
index = np.random.permutation(range(image_count))
light_directions = light_directions[index,:]
images = images[:,:,index]

# write tfrecord
print('\n\n\n')
print(image_count)
print('\n\n\n')
split_num = 100000
nan_buffer = []
for i in range(image_count):
    file_count = i//split_num
    if i%split_num==0:
        if file_count==0:
            pass
        else:
            writer.close()
        writer = tf.python_io.TFRecordWriter('data_tiny/%d.tfrecord'%(file_count))
    image = images[:,:,i]
    image_list = np.reshape(image, [-1])
    image_list = np.float32(image_list)
    nan_num = np.count_nonzero(np.isnan(image_list))
    if nan_num>0:
        print('nan occurs at %d, nan_num: %d' % (i,nan_num))
        nan_buffer.append(image)
        print(image_list)
        time.sleep(3)
        continue
    raw_image = image_list.tostring()
    light = light_directions[i,...]
    raw_light = light.tostring()
    example = tf.train.Example(features = tf.train.Features(feature = {
        'image':_byte_feature(tf.compat.as_bytes(raw_image)),
        'light':_byte_feature(tf.compat.as_bytes(raw_light))
    }))
    writer.write(example.SerializeToString())

writer.close()
nan_buffer = np.array(nan_buffer)
np.save('nan.npy', nan_buffer)
