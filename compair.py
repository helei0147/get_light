import numpy as np
import os,sys
LIGHT_NUM = 5
def regularize(lights):
    block_num = lights.shape[0]
    for i in range(LIGHT_NUM):
        temp = lights[:,i*3:i*3+3]
        length = np.sqrt(np.sum(temp*temp,1))
        length = np.reshape(length, [-1,1])
        length = np.concatenate([length, length, length], 1)
        temp = temp/length
        lights[:,i*3:i*3+3] = temp
    return lights

def main():
    labels = np.load('lights.npy')
    block_num = labels.shape[0]
    est = np.load('p.npy')
    est = np.reshape(est, [-1,LIGHT_NUM*3])
    est = est[0:block_num,:]
    positive_count = np.count_nonzero(est<0)
    print(positive_count)
    print(est[1:5,:])
    print('-------------------')

    est = regularize(est)
    print(est[1:5,:])
    print('-------------------')
    print(labels[1:5,:])

if __name__ == '__main__':
    main()
