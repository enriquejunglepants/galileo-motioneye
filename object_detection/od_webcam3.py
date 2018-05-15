import numpy as np
from PIL import Image
import time

#intializing the web camera device
import cv2

from ODThread import ODThread

od = ODThread()
od.start()

cap = cv2.VideoCapture(0)

cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,480)

ret = True
while (ret):
  ret,image_np = cap.read()
  #print(image_np.size)
  od.set_next(image_np)
  image_np = od.read()

  if image_np is not None:
      cv2.imshow('image',cv2.resize(image_np,(1280,960)))

  if cv2.waitKey(25) & 0xFF == ord('q'):
      od.stop()
      cv2.destroyAllWindows()
      cap.release()
      break
