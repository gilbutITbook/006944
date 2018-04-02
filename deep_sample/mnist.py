import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable,optimizers,Chain

train, test = chainer.datasets.get_mnist(ndim=3)

class Model(Chain):
	def __init__(self):
		super(Model, self).__init__(
			l1=L.Linear(784, 100),
			l2=L.Linear(100, 100),
			l3=L.Linear(100, 10),
		)
	def __call__(self, x):
		h = F.relu(self.l1(x))
		h = F.relu(self.l2(h))
		return self.l3(h)


model = L.Classifier(Model())
optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(model)


batchsize = 1000

def conv(batch,batchsize):
	x=[]
	t=[]
	for j in range(batchsize):
		x.append(batch[j][0])
		t.append(batch[j][1])
	return Variable(np.array(x)),Variable(np.array(t))

for n in range(20):
	for i in chainer.iterators.SerialIterator(train, batchsize,repeat=False):
		x,t = conv(i,batchsize)
         
		model.zerograds()
		loss = model(x,t)
		loss.backward()
		optimizer.update()

		i = chainer.iterators.SerialIterator(test,batchsize, repeat=False).next()
		x,t = conv(i,batchsize)
		loss = model(x,t)
	print n,loss.data
