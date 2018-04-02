import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable,optimizers,Chain

train, test = chainer.datasets.get_mnist(ndim=3)


class Model(Chain):
	def __init__(self):
		super(Model, self).__init__(
			conv1=L.Convolution2D(1, 20, 5),
			conv2=L.Convolution2D(20, 50, 5),
			fc1=L.Linear(800, 500),
			fc2=L.Linear(500, 10),
		)
	def __call__(self, x,train=True):
		cv1 = self.conv1(x)
		relu = F.relu(cv1)
		h = F.max_pooling_2d(relu, 2)
		h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
		h = F.dropout(F.relu(self.fc1(h)), train=train)
		return self.fc2(h)


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
         
	i = chainer.iterators.SerialIterator(test, batchsize).next()
	x,t = conv(i,batchsize)
	loss = model(x,t)
	print n,loss.data
