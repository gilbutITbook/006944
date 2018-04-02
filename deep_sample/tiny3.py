from deel import *
from deel.network import *
from deel.network.googlenet import *
from deel.commands import *
import argparse

deel = Deel()

CNN=GoogLeNet()

parser=argparse.ArgumentParser(description='Image Classifier')
parser.add_argument('--image', '-p', default='test.png', 
			help='image file to classify')
args=parser.parse_args()

CNN.Input(args.image)
CNN.classify()
ShowLabels()
