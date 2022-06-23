"""OpenCV deep fryer attempt."""
from cv2 import INTER_AREA
import numpy as np
import cv2 as cv

img = cv.imread("dogphotos/dog.jpg")
amount = int(input("how intense (out of 255)?"))
for i in range(len(img)):
    for k in range(len(img[i])): #img[i][k] contains an RGB array for the pixel i,k. eg: img[10][96] = [0,255,0] would be a green pixel at pos 10,96
        for j in range(len(img[i][k])):      
            img[i][k][j] = round(img[i][k][j]/amount) * amount  #round each pixel to the nearest ___
            deviation = np.random.randint(0,amount/3) # add some noise
            if np.random.randint(0,1)>1:
                deviation *= -1
            img[i][k][j] += deviation  

img = cv.resize(img, (1080,1080), 1, 1, interpolation=INTER_AREA)
cv.imshow('image',img)
keyboard = cv.waitKey(0)