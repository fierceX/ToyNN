import random
import numpy as np
import matplotlib.pyplot as plt
import sys
from io import BytesIO
import gzip
import struct
import NNlayer

def read_data(label_url, image_url):
    with gzip.open(label_url) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(image_url, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(
            len(label), rows, cols)
    return (label, image)

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -1/m*np.sum(np.multiply(Y,np.log(AL))+np.multiply((1-Y),np.log(1-AL)))
    cost = np.squeeze(cost)
    delta = AL - Y
    return cost,delta.T

def onehot(label):
    oh = []
    for l in label:
        le = [ (int)(x == l) for x in range(10)]
        oh.append(le)
    return np.array(oh)

(train_lable, train_image) = read_data(
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz')

def data_iter(data,label,bs):
    idx = list(range(label.shape[0]))
    random.shuffle(idx)
    for i in range(0,data.shape[0],bs):
        j = np.array(idx[i:min(i+bs,label.shape[0])])
        yield np.take(data,j,axis=0),np.take(label,j)
net = NNlayer.NetModel()
net.add(NNlayer.dens(784,30,'dens1'))
net.add(NNlayer.activate(model='relu',name='relu'))
net.add(NNlayer.dens(30,10,'dens2'))
net.add(NNlayer.activate(model='sigmoid',name='sigmoid'))


for i in range(1):
    cost = 0.0
    for data, label in data_iter(train_image,train_lable,20):
        data = data.astype('float32') / 255
        data = np.reshape(data,(20,-1))
        l = onehot(label)
        out = net.forward(data.T)
        minicost,delta = compute_cost(out.T,l)
        cost += minicost
        net.backward(delta)
        net.update_params(0.05)
    print(cost/len(train_lable))

net.save_params("net.params")