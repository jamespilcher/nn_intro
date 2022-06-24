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



X_train, X_test = X_train.reshape(60000,-1) / 255, X_test.reshape(10000,-1) / 255
print(X_train.shape)
X_train, X_test = X_train.T, X_test.T #OME BACK TO THIS!
y_train, y_test = zero_detectorfy(y_train).reshape(1,60_000), zero_detectorfy(y_test).reshape(1,10_000)

#Z = np.matmul(W.T, X_train) + b


#A = sigmoid(Z)

if True:
    ################################################ ## 

    learning_rate = 1

    X = X_train
    Y = y_train
    print(X.shape)
    n_x = X.shape[0]
    m = X.shape[1]

    W = np.random.randn(n_x, 1) * 0.01
    print(Y)
    b = np.zeros((1, 1))

    for i in range(5000):
        Y_hat = sigmoid(np.matmul(W.T, X) + b) # the prediction
        cost = compute_loss(Y, Y_hat)          # the cross entropy cost

        dW = (1/m) * np.matmul(X, (Y_hat - Y).T)   # dLoss/dWeight (minimising cross entropy with respect to weight)
        db = (1/m) * np.sum(Y_hat - Y, axis=1, keepdims=True) # dLoss/dbias (minimising cross entropy with respect to bias)

        W = W - learning_rate * dW # adjust the weight
        b = b - learning_rate * db # adjust the bias
        if (i % 100 == 0):
            print("Epoch", i, "cost: ", cost)
            print("Y_hat: ", Y_hat)

    print("Final cost:", cost)
    np.save("W.npy", W)
    np.save("b.npy", b)
    ###################################################
W = np.load("W.npy")
b = np.load("b.npy")
print("Testing...")
print(y_test.shape)
print(X_test.shape)

correct = 0
incorrect = 0
for k in range(10000):
    Y_hat = sigmoid(np.matmul(W.T, X_test) + b)  #1x10000 predictions...
    fail = False
    if y_test[0,k] == 1:
        if (Y_hat[0, k] > 0.5):
            correct += 1
        if (Y_hat[0, k] < 0.5):
            incorrect += 1
            incorrectnumber = ("this is a zero, you said it wasn't")
            fail = True
    else:
        if (Y_hat[0, k] < 0.5):
            correct += 1
        if (Y_hat[0, k] > 0.5):
            incorrect += 1
            incorrectnumber = ("this is not a zero, you said it was")
            fail = True

    if False:
        img = X_test[:,k].reshape(28,28)
        img = cv.resize(img, (1080,1080), 1, 1, interpolation=INTER_AREA)
        cv.imshow(incorrectnumber,img)
        keyboard = cv.waitKey(0)
print("Accuracy of: ", correct/10000)