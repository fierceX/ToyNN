import random
import numpy as np
from NNlayer import *
from tool import *
from model import get_net

(test_lable, test_image) = read_data('./t10k-labels-idx1-ubyte.gz', './t10k-images-idx3-ubyte.gz')

net = get_net()
net.load_params('net.params')

tloss,acc =  get_loss_accuracy(net,test_image,test_lable,128)
print('Test Loss: %f Test Acc: %f' % (tloss,acc))