import cv2
from deel import *
from deel.network import *
from deel.commands import *

deel = Deel()

CNN = GoogLeNet()

vid = cv2.VideoCapture("test.mp4")

while True:
   ret, img = vid.read()
   CNN.Input(img)
 
   CNN.classify()
   ShowLabels()
    
   cv2.imshow('vid', img)
   if cv2.waitKey(10) > 0:
       break

vid.release()
cv2.destroyAllWindows()
