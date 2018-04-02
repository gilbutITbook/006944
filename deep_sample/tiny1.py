from deel import *
from deel.network import *
from deel.commands import *

deel = Deel()

CNN=Alexnet()

CNN.Input("deel.png")
CNN.classify()
ShowLabels()
