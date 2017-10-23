import os,sys
import numpy as np

def main():
    for model_index in range(5):
        for material_index in range(0,100,5):
            save_npy(model_index,material_index)
            
def save_npy(model_index,material_index):
    folder_name = 'Results_small/'+str(material_index)+'_'+str(model_index)+'/'
    image_buffer = []
    pixel_count = 0
    observations = []
    for i in range(200):
        rgbfile = open(folder_name+str(i) + '.rgb', 'rb')
        temp = np.fromfile(rgbfile,dtype = np.float32, count = -1);
        pixel_count = len(temp)//3;
        image_buffer.append(temp)
    print(pixel_count)
    observations = np.transpose(image_buffer)
    np.save(str(model_index)+'_'+str(material_index)+'.npy',observations)


if __name__ == '__main__':
    main()
