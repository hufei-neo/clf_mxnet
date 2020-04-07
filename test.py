from models.Clfs import Textcnn

model = Textcnn(20000, 300, 100, 50, 0.2,[1024,238])
print(model)

from mxnet import nd
from mxnet.gluon import nn

nd.one_hot()



