import random
import numpy as np
import os
import pickle

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A

def relu(Z):
    A = np.maximum(0,Z)
    return A

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA,Z):
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

class NNlayer():
    def __init__(self,name):
        self.name =  name
    def set_params(self,params):
        pass
    def forward(self,X):
        return X
    def backward(self,dA):
        return dA
    def update_params(self,learning_rate):
        pass
    def params(self):
        return None
    def load_params(self,params):
        pass

class dens(NNlayer):
    def __init__(self,n_in,n_out,name):
        super(dens,self).__init__(name)
        self.W = np.random.randn(n_out,n_in)*0.01
        self.b = np.zeros((n_out,1))
    def load_params(self,params):
        self.W = params[0]
        self.b = params[1]
    def forward(self,X):
        self.A = X
        return np.dot(self.W,X)+self.b
    def backward(self,dZ):
        m = self.A.shape[1]
        self.dW = 1/m*np.dot(dZ,self.A.T)
        self.db = 1/m*np.sum(dZ,axis=1,keepdims=True)
        self.dA = np.dot(self.W.T,dZ)
        return self.dA
    def update_params(self,learning_rate):
        self.W = self.W - learning_rate * self.dW
        self.b = self.b - learning_rate * self.db
    def params(self):
        return [self.W,self.b]

class activate(NNlayer):
    def __init__(self,model,name):
        super(activate,self).__init__(name)
        self.model = model
    def forward(self,X):
        self.A = X
        if self.model == 'sigmoid':
            return sigmoid(X)
        elif self.model == 'relu':
            return relu(X)
    def backward(self,dA):
        if self.model == 'sigmoid':
            return sigmoid_backward(dA,self.A)
        elif self.model == 'relu':
            return relu_backward(dA,self.A)

class NetModel():
    def __init__(self):
        self.net=[]
    def add(self,n):
        self.net.append(n)
    def forward(self,X):
        a = X
        for n in self.net:
            a = n.forward(a)
        return a
    def backward(self,dA):
        a = dA
        for i in range(1,len(self.net)+1):
            a = self.net[-i].backward(a)
        return a
    def update_params(self,lr):
        for n in self.net:
            n.update_params(lr)
    def save_params(self,name):
        f = open(name,"wb")
        for n in self.net:
            if n.params() is not None:
                pickle.dump(n.params(),f)
        f.close()
    def load_params(self,name):
        f = open(name,"rb")
        for n in self.net:
            if n.params() is not None:
                params = pickle.load(f)
                n.load_params(params)
        f.close()
    