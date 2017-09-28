import tensorflow as tf
import numpy as np
tf.InteractiveSession()
feature = {'image': tf.FixedLenFeature([1024], tf.float32),
           'light': tf.FixedLenFeature([3], tf.float32)}

filename_queue = tf.train.string_input_producer(['data.tfrecord'], num_epochs = 1)

reader = tf.TFRecordReader()

_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(serialized_example, features = feature)

image = tf.cast(features['image'], tf.float32)
light = tf.cast(features['light'], tf.float32)

image = tf.reshape(image,[32,32])

image.eval()
