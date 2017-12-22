import random
import numpy as np
from NNlayer import *
from tool import *
from model import get_net

(train_lable, train_image) = read_data('./train-labels-idx1-ubyte.gz', './train-images-idx3-ubyte.gz')

(test_lable, test_image) = read_data('./t10k-labels-idx1-ubyte.gz', './t10k-images-idx3-ubyte.gz')

net = get_net()

for i in range(15):
    cost = 0.0
    for data, label in data_iter(train_image,train_lable,20):
        data = data.astype('float32') / 255
        data = np.reshape(data,(data.shape[0],data.shape[1],data.shape[2],1))
        l = onehot(label)
        out = net.forward(data)
        minicost,delta = compute_cost(out.T,l)
        cost += minicost
        net.backward(delta)
        net.update_params(0.1)
    tloss,acc =  get_loss_accuracy(net,test_image,test_lable,128)
    print("Epoch %d. Train Loss: %f Test Loss: %f Test Acc: %f" % (i,cost/len(train_lable),tloss,acc))

net.save_params("net.params")