"""Neural Network from scratch"""
from zlib import Z_BLOCK
from cv2 import INTER_AREA
import numpy as np
import cv2 as cv
import tensorflow.keras as keras
(X_train,y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


def zero_detectorfy_brute(y): #Will make the groundtruth a 1 if its a 0, otherwise groundTruth = 0 
    y_new = np.zeros(y.shape) # re
    for i in range(len(y)):
        if y[i] == 0:
            y_new[i] = 1
    return y_new

def zero_detectorfy(y): #Will make the groundtruth a 1 if its a 0, otherwise groundTruth = 0 
    y_new = np.zeros(y.shape) # re
    y_new[np.where(y == 0)] = 1 #np.where will return the index where the condition is satisfied in the given np array.
    return y_new

def sigmoid(z):
    s = 1 / (1+np.exp(-z))
    return s

def compute_loss(Y, Y_hat):
    m = Y.shape[1]
    L = -(1/m) * ( np.sum(np.multiply(np.log(Y_hat),Y)) + np.multiply(np.log(1-Y_hat),(1-Y)) )
    return L



X_train, X_test = X_train.reshape(60000,1,-1) / 255, X_test.reshape(10000,1,-1) / 255
print(X_train.shape)
X_train, X_test = X_train.T, X_test.T #OME BACK TO THIS!
y_train, y_test = zero_detectorfy(y_train).reshape(1,60_000), zero_detectorfy(y_test).reshape(1,10_000)

#Z = np.matmul(W.T, X_train) + b



ele = np.random.randint(0,9999)
ground_truth = y_train[0,ele]
number_image = X_train[ele]
number_image = cv.resize(number_image, (1080,1080), 1, 1, interpolation=INTER_AREA)
cv.imshow(str(ground_truth),number_image)
keyboard = cv.waitKey(0)
