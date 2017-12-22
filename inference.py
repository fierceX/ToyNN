import sys
from model import get_net
import cv2
import numpy as np

path = sys.argv[1]
net = get_net()
net.load_params('net.params')

img = cv2.imread(path,0)
img = cv2.resize(img,(28,28))
data = img.astype('float32')/255
data = np.reshape(data,(1,28,28,1))

out = net.forward(data)
print(out.argmax(axis=0)[0])
