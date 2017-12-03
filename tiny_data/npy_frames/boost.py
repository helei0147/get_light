# encoding=utf-8
# please use python 2
import cv2
import numpy as np
npy_file_num = 5
for i in range(npy_file_num):
    frames = np.load('%d.npy'%(i))
    block_num = frames.shape[0]
    new_block_buffer = []
    for block_index in range(block_num):
        block = frames[block_index, :, :, :]
        frame_num, height, width = block.shape
        canvas = np.zeros([height, width])
        generated = []
        # 生成两帧之间的差
        for sub in range(frame_num-1):
            canvas = block[sub+1, :, :] - block[sub, :, :]
            generated.append(canvas)
        # sobel 算子生成梯度图
        for frame in range(frame_num):
            sobelx = cv2.Sobel(block[frame,:,:], cv2.CV_64F, 1, 0,ksize=5)
            generated.append(sobelx)
        for frame in range(frame_num):
            sobely = cv2.Sobel(block[frame,:,:], cv2.CV_64F, 0, 1,ksize=5)
            generated.append(sobely)
        generated = np.array(generated)
        new_block = np.concatenate((block,generated), axis=0)
        new_block_buffer.append(new_block)
    new_block_buffer = np.array(new_block_buffer)
    np.save('new_%d.npy'%(i),new_block_buffer)
