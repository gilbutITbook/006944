# -*- coding: utf-8 -*-

import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable,optimizers,Chain,cuda
import data
import cPickle as pickle

mnist = data.load_mnist_data()
x_all = mnist['data'].astype(np.float32) / 255
y_all = mnist['target'].astype(np.int32)
x_train, x_test = np.split(x_all, [60000])
y_train, y_test = np.split(y_all, [60000])

class Model(Chain):
  def __init__(self):
    super(Model, self).__init__(
      l1=L.Linear(784, 400),
      l1b=L.Linear(400, 784),
      l2=L.Linear(400, 100),
      l2b=L.Linear(100, 400),
      l3=L.Linear(100, 49),
      l3b=L.Linear(49, 100),
      l4=L.Linear(49, 16),
      l4b=L.Linear(16, 49),
      l5=L.Linear(16, 3),
      l5b=L.Linear(3, 16),
      )
  def __call__(self, x,train=True,layer=0):
    tf = [False]
    for i in range(6):
      tf.append(False if i<layer-1 else train)


    h = F.dropout(F.relu(self.l1(x)),train=tf[1])
    if layer==1:
      return F.dropout(self.l1b(h),train=train),x
    x = h
    h = F.dropout(F.relu(self.l2(x)),train=tf[2])
    if layer==2:
      return F.dropout(self.l2b(h),train=train),x
    x = h
    h = F.dropout(F.relu(self.l3(x)),train=tf[3])
    if layer==3:
      return F.dropout(self.l3b(h),train=train),x
    x = h
    h = F.dropout(F.relu(self.l4(x)),train=tf[4])
    if layer==4:
      return F.dropout(self.l4b(h),train=train),x
    x = h
    h = F.dropout(F.relu(self.l5(x)),train=tf[5])
    if layer==5:
      return F.dropout(self.l5b(h),train=train),x
    return h

  def dump(self):
    pickle.dump(self.l1,open('l1.pkl', 'w'))
    pickle.dump(self.l2,open('l2.pkl', 'w'))
    pickle.dump(self.l3,open('l3.pkl', 'w'))
    pickle.dump(self.l4,open('l4.pkl', 'w'))
    pickle.dump(self.l5,open('l5.pkl', 'w'))

model = Model()

gpu=-1 #don't use gpu
if gpu>=0:
  cuda.get_device(gpu).use()
  xp = cuda.cupy
  model.to_gpu(gpu)
else:
 xp=np

optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)

batchsize = 100
datasize = 60000
epochs = 1

model.dump()
import matplotlib.pyplot as plt
for layer in range(1,6):
  optimizer.setup(model)
  for j in range(epochs):
    indexes = np.random.permutation(datasize)
    for i in range(0, datasize, batchsize):
      x = Variable(xp.asarray(x_train[indexes[i : i + batchsize]]))
      t = Variable(xp.asarray(y_train[indexes[i : i + batchsize]]))
      model.zerograds()
      img,x = model(x,layer = layer)
      loss = F.mean_squared_error(img, x)
      loss.backward()
      optimizer.update()

    x = Variable(xp.asarray(x_test))
    img,x = model(x,train=False,layer=layer)
    loss = F.mean_squared_error(img, x)
    print ("layer:",layer,j,loss.data)

    """if layer==1: #층의 가시화
      if j%10==0:
      img = img.data[0].reshape(28,28)
      plt.imshow(img)
      plt.savefig("encoded_%d.png"%j)
    if layer==2:
      if j%10==0:
        img = img.data[0].reshape(20,20)
        plt.imshow(img)
        plt.savefig("encoded_%d.png"%j)
        """
  model.dump()
