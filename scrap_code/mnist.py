from cv2 import INTER_AREA, ROTATE_180
import numpy as np
import cv2 as cv
import tensorflow.keras as keras
training, testing = keras.datasets.mnist.load_data()

ele = np.random.randint(0,9999)
firstEl = training[0][ele]
firstEl = cv.rotate(firstEl,rotateCode=ROTATE_180)
groundTruth = training[1][ele]
firstEl = cv.resize(firstEl, (1080,1080), 1, 1, interpolation=INTER_AREA)
#firstEl[firstEl > 20] = 255
cv.imshow(str(groundTruth),firstEl)
keyboard = cv.waitKey(0)
