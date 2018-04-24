import numpy as np
from matplotlib import pyplot

def cal_deg(logits, gt):
    batch_size, light_num = logits.shape
    light_num = int(light_num/3)
    logits = normalize(logits)
    logits_buffer = np.split(logits, light_num, axis=1)
    gt_buffer = np.split(gt, light_num, axis=1)
    deg_buffer = []
    for i in range(light_num):
        a = logits_buffer[i]
        b = gt_buffer[i]
        to_arccos = np.sum(a*b,1)
        to_arccos[to_arccos>1]=1
        to_arccos[to_arccos<-1] = -1
        deg = np.arccos(to_arccos)/np.pi*180
        deg = np.reshape(deg, [batch_size, 1])
        deg_buffer.append(deg)
    deg_buffer = np.concatenate(deg_buffer,1)
    return deg_buffer    
    
def normalize(logits):
    group_num, light_num = logits.shape
    if light_num%3!=0:
        return [-1]
    light_num = int(light_num/3)
    for i in range(light_num):
        logits[:,i*3:i*3+3] = normalize_vec(logits[:,i*3:i*3+3])
    return logits

def normalize_vec(vec):
    vec_num = vec.shape[0]
    length = np.sum(vec*vec,1)
    length = np.sqrt(length)
    length = np.reshape(length, [vec_num,1])
    length = np.concatenate([length, length, length],1)
    vec = vec/length
    return vec

if __name__ == '__main__':
    labels = np.load('np_labels.npy')
    logits = np.load('np_logits.npy')
    batch_num, batch_size, light_dims = labels.shape
    deg_buffer = []
    for i in range(batch_num):
        degs = cal_deg(logits[i], labels[i])
        deg_buffer.append(degs)
    deg_buffer = np.array(deg_buffer)
    print(deg_buffer.shape)

    temp = np.sum(deg_buffer,2)
    line = np.reshape(temp, [-1])
    pyplot.hist(line, 200)
    avg = np.average(temp)
    print('avg: ', avg)
    med = np.median(line)
    print('med: ', med)
    np.save('degs.npy', np.array(deg_buffer))