from NNlayer import *


def get_net():

    # net.add(Conv2D([3,3,1],16,0,2,'conv1'))
    # net.add(Pool(2,2,mode='max',name='pool1'))
    # net.add(activate(model='relu',name='relu'))
    # net.add(Flatten("flatten"))
    # net.add(dens(576,10,'dens1'))
    # net.add(activate(model='sigmoid',name='sigmoid'))


    net = NetModel()
    net.add(Flatten("flatten"))
    net.add(dense(784,128,'dens1'))
    net.add(activate(model='relu',name='relu'))
    net.add(dense(128,10,'dens2'))
    net.add(activate(model='sigmoid',name='sigmoid'))
    return net