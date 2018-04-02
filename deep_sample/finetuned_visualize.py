import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable,optimizers,Chain
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
    )
  def forward(self, x):
    x = self.l1(x)
    x = F.relu(self.l2(x))
    x = F.relu(self.l3(x))
    x = F.relu(self.l4(x))
    h = self.l5(x)
    return h
  def load(self):
    self.l1 = pickle.load(open('l1f.pkl', 'r'))
    self.l2 = pickle.load(open('l2f.pkl', 'r'))
    self.l3 = pickle.load(open('l3f.pkl', 'r'))
    self.l4 = pickle.load(open('l4f.pkl', 'r'))
    self.l5 = pickle.load(open('l5f.pkl', 'r'))


model = Model()
model.load()
import matplotlib.pyplot as plt

batchsize = 100
datasize = 60000
x = np.random.rand(datasize)
y = np.random.rand(datasize)
z = np.random.rand(datasize)
cl = np.random.rand(datasize)
from mpl_toolkits.mplot3d.axes3d import Axes3D
print "start"
cnt=0
for i in range(0,datasize, batchsize):
  indexes = np.random.permutation(datasize)
  img = Variable(x_train[indexes[i : i + batchsize]])
  t = y_train[indexes[i : i + batchsize]]
  i = model.forward(img)
  for v,c in zip(i.data,t):
    print v
    x[cnt] = v[0]
    y[cnt] = v[1]
    z[cnt] = v[2]

    cl[cnt] = np.float(c)
    cnt+=1

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter3D(x, y,z, c=cl)
plt.show()
