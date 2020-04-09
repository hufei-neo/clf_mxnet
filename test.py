from models.Clfs import Textcnn

model = Textcnn(20000, 300, 100, 50, 0.2,[1024,238])
print(model)

# from mxnet import nd
# from mxnet.gluon import nn
#
# nd.one_hot()
#
# pred = nd.argmax()
# from mxnet.gluon import loss as gloss, nn
# loss = gloss.SoftmaxCrossEntropyLoss()
# loss()


from mxnet.gluon import loss as gloss, nn, Trainer
from mxnet import autograd
import time

loss = gloss.SoftmaxCrossEntropyLoss()



