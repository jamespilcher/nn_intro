"""Neural Network from scratch"""
from zlib import Z_BLOCK
from cv2 import INTER_AREA
import numpy as np
import cv2 as cv
import tensorflow.keras as keras
(X_train,y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

def one_hot(y):
    y_new = np.zeros((y.size, y.max()+1 ))
    for i in range(len(y)):
        y_new[i][y[i]] = 1
    return y_new.reshape(-1, y.size)

def zero_detectorfy_brute(y): #Will make the groundtruth a 1 if its a 0, otherwise groundTruth = 0 
    y_new = np.zeros(y.shape) # re
    for i in range(len(y)):
        if y[i] == 0:
            y_new[i] = 1
    return y_new

def zero_detectorfy(y): #Will make the groundtruth a 1 if its a 0, otherwise groundTruth = 0 
    y_new = np.zeros(y.shape) # re
    y_new[np.where(y > 0)] = 1 #np.where will return the index where the condition is satisfied in the given np array
    return y_new.astype(int)

def sigmoid(z):
    s = 1 / (1+np.exp(-z))
    return s

def compute_loss(Y, Y_hat):
    m = Y.shape[1]
    L = -(1/m) * ( np.sum(np.multiply(np.log(Y_hat),Y)) + np.multiply(np.log(1-Y_hat),(1-Y)) )

    return L

def compute_loss_v2(Y, Y_hat):
    m = Y.shape[1]
    L = -(1/m) * (np.sum(np.multiply(Y, np.log(Y_hat))))
    return L

X_train, X_test = X_train.reshape(60000,-1) / 255, X_test.reshape(10000,-1) / 255
X_train, X_test = X_train.T, X_test.T #OME BACK TO THIS!
#y_train, y_test = one_hot(zero_detectorfy(y_train)), one_hot(zero_detectorfy(y_test))

y_train, y_test = zero_detectorfy(y_train), zero_detectorfy(y_test)

#Z = np.matmul(W.T, X_train) + b


#A = sigmoid(Z)

if True:
    ################################################## 

    learning_rate = 1

    X = X_train
    Y = y_train
    n_x = X.shape[0]
    m = X.shape[1]

    W = np.random.randn(n_x, 2) * 0.01
    b = np.zeros((1, 1))

    for i in range(1500):
        Y_hat = sigmoid(np.matmul(W.T, X) + b) # the prediction
        cost = compute_loss_v2(Y, Y_hat)          # the cross entropy cost
####################
        print("Y_hat shape:", Y_hat.shape)
        print("Y shape:", (Y).shape)

        print("(np.matmul(Y, (Y - Y_hat).T):", (np.matmul(Y, (Y - Y_hat)).shape))

        print("X shape:", X.shape)
        dLdZ = Y - np.matmul(Y, (Y_hat))
        dW = (1/m) * np.matmul(X, (np.matmul(Y, (Y - Y_hat).T)))   # dLoss/dWeight (minimising cross entropy with respect to weight)
        db = (1/m) * np.sum(np.matmul(Y, (Y - Y_hat)), axis=1, keepdims=True) # dLoss/dbias (minimising cross entropy with respect to bias)

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
for k in range(200):
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
print("Accuracy of: ", correct/2000)