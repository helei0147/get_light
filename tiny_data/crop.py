import os,sys
import time
import numpy as np
from scipy import misc

for light_index in range(200):
    light_colle = []
    info_colle = []
    for model_index in range(5):
        mask = misc.imread('mask/%d.png'% model_index)
        mask = mask>0
        [height, width] = mask.shape
        #channels = np.ndarray([height,width,3], dtype = np.float32)
        #mat_buffer = np.ndarray([height, width, 20], dtype = np.float32)
        channels = np.zeros([height,width,3])
        channels = np.float32(channels)
        mat_buffer = np.zeros([height, width, 20])
        mat_buffer = np.float32(mat_buffer)
        channel_counter = 0
        for material_index in range(0,100,5):
            folder_name = 'Results_small/%d_%d/' % (material_index, model_index)
            filename = folder_name+str(light_index)+'.rgb'
            rgb = np.fromfile(filename, dtype = np.float32, count = -1);
            channels[mask] = np.reshape(rgb,[-1,3])
            gs_channel = channels[:,:,0]*0.2989+channels[:,:,1]*0.5870+channels[:,:,2]*0.1140
            mat_buffer[:,:,channel_counter] = gs_channel
        nan_num = np.count_nonzero(np.isnan(mat_buffer))
        if nan_num>0:
            print('nan occurs at model: %d, light: %d'%(model_index, light_index))
            print('nan_num: %d'%(nan_num))
            time.sleep(3)
        pos_buffer = np.load('mask/pos_buffer/%d.npy'%model_index)
        crop_number = pos_buffer.shape[0]
        for i in range(crop_number):
            top = pos_buffer[i, 0];
            bottom = pos_buffer[i, 1];
            left = pos_buffer[i, 2];
            right = pos_buffer[i, 3];
            crop_batch = mat_buffer[top:bottom, left:right, :]
            light_colle.append(crop_batch.tolist())
            info_colle.append([model_index, i])
    light_colle = np.array(light_colle, dtype = np.float32)
    info_colle = np.array(info_colle, dtype = np.float32)
    print(light_colle.shape)
    print(type(light_colle[0,0,0,0]))
    np.save('light_npy/%d.npy'%light_index, light_colle)
    np.save('info_npy/%d.npy'%light_index, info_colle)
