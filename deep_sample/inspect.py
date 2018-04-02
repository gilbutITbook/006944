#inspect.py
import argparse
import sys
import numpy as np
from PIL import Image
import cPickle as pickle
import chainer
import math
import chainer.functions as F
import chainer.links as L
from chainer import serializers

parser = argparse.ArgumentParser(
	description='Image inspection using chainer')
parser.add_argument('image', help='Path to inspection image file')
parser.add_argument('--model', default='model.h5',
			help='Path to the mean file (computed by compute_mean.py)')
parser.add_argument('--mean', default='mean.npy',
			help='Path to the mean file (computed by compute_mean.py)')
args = parser.parse_args()


def read_image(path):
	image = np.asarray(Image.open(path)).transpose(2, 0, 1)
	top = left = cropwidth / 2
	bottom = model.insize + top
	right = model.insize + left
	image = image[:, top:bottom, left:right].astype(np.float32)
	image -= mean_image[:, top:bottom, left:right]
	image /= 255
	return image

import network

mean_image = pickle.load(open(args.mean, 'rb'))

model = network.Network()
serializers.load_hdf5(args.model, model)
cropwidth = 256 - model.insize
model.to_cpu()
model.train=False

img = read_image(args.image)

score = model.predict(np.asarray([img]))

categories = np.loadtxt("labels.txt", str, delimiter="\t")

top_k = 20
prediction = zip(score.data[0].tolist(), categories)
prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)
for rank, (score, name) in enumerate(prediction[:top_k], start=1):
	print('#%d | %s | %4.1f%%' % (rank, name, score * 100))
