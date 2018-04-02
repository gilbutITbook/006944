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
      l2=L.Linear(400, 100),
      l3=L.Linear(100, 49),
      l4=L.Linear(49, 16),
      l5=L.Linear(16, 3),
      l6=L.Linear(3, 10),
      )
  def __call__(self, x,train=True):
    h = self.l1(x)
    h = F.dropout(F.relu(self.l2(h)),train=train)
    h = F.dropout(F.relu(self.l3(h)),train=train)
    h = F.dropout(F.relu(self.l4(h)),train=train)
    h = self.l5(h)
    h = self.l6(h)
    return h
  def dump(self):
    pickle.dump(self.l1,open('l1f.pkl', 'w'))
    pickle.dump(self.l2,open('l2f.pkl', 'w'))
    pickle.dump(self.l3,open('l3f.pkl', 'w'))
    pickle.dump(self.l4,open('l4f.pkl', 'w'))
    pickle.dump(self.l5,open('l5f.pkl', 'w'))
    pickle.dump(self.l6,open('l6f.pkl', 'w'))
  def load(self):
    pickle.dump(self.l1,open('l1.pkl', 'w'))
    pickle.dump(self.l2,open('l2.pkl', 'w'))
    pickle.dump(self.l3,open('l3.pkl', 'w'))
    pickle.dump(self.l4,open('l4.pkl', 'w'))
    pickle.dump(self.l5,open('l5.pkl', 'w'))
    pickle.dump(self.l6,open('l6.pkl', 'w'))

model = Model()
model.load()
optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(model)
batchsize = 100
datasize = 1000
epochs=500
 
cmodel = L.Classifier(model)
for e in range(epochs):
  for i in range(0,datasize, batchsize):
    indexes = np.random.permutation(datasize)
    img = Variable(x_train[indexes[i : i + batchsize]])
    t = Variable(y_train[indexes[i : i + batchsize]])
    optimizer.update(cmodel, img, t)

  print e,cmodel.loss.data,cmodel.accuracy.data

  model.dump()
