# encoding=utf-8
# please use python 2
import cv2
import numpy as np
from scipy import misc
LIGHT_NUMBER = 5
npy_file_num = 40
for i in range(npy_file_num):
    frames = np.load('%d.npy'%(i))
    block_num = frames.shape[0]//LIGHT_NUMBER
    new_block_buffer = []
    counter = 0
    for block_index in range(block_num):
        counter = counter+1
        block = frames[block_index*LIGHT_NUMBER:block_index*LIGHT_NUMBER+LIGHT_NUMBER, :, :]
        frame_num, height, width = block.shape
        canvas = np.zeros([height, width])
        generated = []
        # sobel 算子生成梯度图
        for frame in range(frame_num):
            sobelx = cv2.Sobel(block[frame,:,:], cv2.CV_64F, 1, 0,ksize=5)
            generated.append(sobelx)
        for frame in range(frame_num):
            sobely = cv2.Sobel(block[frame,:,:], cv2.CV_64F, 0, 1,ksize=5)
            generated.append(sobely)
        generated = np.array(generated)
        new_block = np.concatenate((block,generated), axis=0)
        print('new_block shape: ', new_block.shape)
        # canv = new_block
        # h_b_ori = np.max(canv[0:5,:,:])
        # canv[0:5,:,:] = canv[0:5,:,:]/h_b_ori*255
        # h_b_minus = np.max(canv[5:9,:,:])
        # canv[5:9,:,:] = canv[5:9,:,:]*255
        # h_b_x = np.max(canv[9:14,:,:])
        # canv[9:14,:,:] = canv[9:14,:,:]/h_b_x*255
        # h_b_y = np.max(canv[14:,:,:])
        # canv[14:,:,:] = canv[14:,:,:]/h_b_y*255
        # canv = np.concatenate(new_block, axis=0)
        # canv_name = 'fuck/%d_%d.png'%(i, counter)
        # misc.imsave(canv_name, canv)
        new_block_buffer.append(new_block.astype(np.float32))
    new_block_buffer = np.array(new_block_buffer)
    print('new_block_buffer.shape:', new_block_buffer.shape)
    np.save('new_%d.npy'%(i),new_block_buffer)
    print('file new_%d.npy finished'%(i))
