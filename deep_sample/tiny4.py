from deel import *
from deel.network import *
from deel.commands import *

deel = Deel()
nin = NetworkInNetwork()

InputBatch(train="data/train.txt",
		val="data/test.txt")

def workout(x,t):
	nin.classify(x)
	return nin.backprop(t)

BatchTrain(workout)

