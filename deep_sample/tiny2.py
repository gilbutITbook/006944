from deel import *
from deel.network import *
from deel.network.googlenet import *
from deel.commands import *

deel = Deel()

CNN=GoogLeNet()

CNN.Input("test.png")
CNN.classify()
ShowLabels()
