from cv2 import CV_16S
import numpy as np
import cv2 as cv

img = cv.imread("trafficphotos/trafficA.png")
img2 = cv.imread("trafficphotos/trafficB.png")

#img = np.delete(img, 3, 2)
#img2 = np.delete(img2, 3, 2)

minus = np.subtract(img2,img)
add = np.add(img,img2)
result = np.absolute(minus)
cv.imshow('image',result)
keyboard = cv.waitKey(30)

