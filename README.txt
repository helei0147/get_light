先运行crop.py, 把Result文件夹中的rgb文件根据mask转换为batch
运行generate_tfrecord.py 把上面生成的light_npy文件夹中的按光照分类的npy转换为tfrecord，方便读取。
