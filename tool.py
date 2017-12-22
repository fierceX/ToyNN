import random
import numpy as np
import sys
from io import BytesIO
import gzip
import struct

def read_data(label_url, image_url):
    with gzip.open(label_url) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(image_url, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(
            len(label), rows, cols)
    return (label, image)

def onehot(label):
    oh = []
    for l in label:
        le = [ (int)(x == l) for x in range(10)]
        oh.append(le)
    return np.array(oh)

def data_iter(data,label,bs):
    idx = list(range(label.shape[0]))
    random.shuffle(idx)
    for i in range(0,data.shape[0],bs):
        j = np.array(idx[i:min(i+bs,label.shape[0])])
        yield np.take(data,j,axis=0),np.take(label,j)

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -1/m*np.sum(np.multiply(Y,np.log(AL))+np.multiply((1-Y),np.log(1-AL)))
    cost = np.squeeze(cost)
    delta = AL - Y
    return cost,delta.T

def get_loss_accuracy(net,data,label,bs):
    cost = 0.0
    acc=0.0
    n = 0
    for i,(d,l) in enumerate(data_iter(data,label,bs)):
        d = d.astype('float32') / 255
        d = np.reshape(d,(d.shape[0],d.shape[1],d.shape[2],1))
        lone = onehot(l)
        out = net.forward(d)
        minicost,_ = compute_cost(out.T,lone)
        cost += minicost
        acc += accuracy(out,l)
        n = i
    return cost/len(label),acc/n

def accuracy(output,label):
    return np.mean(output.argmax(axis=0) == label)