"""Neural Network from scratch"""
from zlib import Z_BLOCK
from cv2 import INTER_AREA
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import cv2 as cv
import tensorflow.keras as keras
(X_train,y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

def zero_detectorfy(y): #Will make the groundtruth a 1 if its a 0, otherwise groundTruth = 0 
    y_new = np.zeros(y.shape) # re
    y_new[np.where(y > 0)] = 1 #np.where will return the index where the condition is satisfied in the given np array
    return y_new.astype(int)

def one_hot(y):
    y_new = np.zeros((y.size, y.max()+1 ))
    for i in range(len(y)):
        y_new[i][y[i]] = 1
    return y_new.reshape(10,-1)


def sigmoid(z):
    s = 1 / (1+np.exp(-z))
    return s

def compute_loss(Y, Y_hat):
    m = Y.shape[1]
    L = -(1/m) * (np.sum(np.multiply(Y, np.log(Y_hat))))
    return L

def feed_forward(X, params):

    cache = {}

    cache["Z1"] = np.matmul(params["W1"], X) + params["b1"]
    cache["A1"] = sigmoid(cache["Z1"])
    cache["Z2"] = np.matmul(params["W2"], cache["A1"]) + params["b2"]
    cache["A2"] = np.exp(cache["Z2"]) / np.sum(np.exp(cache["Z2"]), axis=0)

    return cache

def back_propagate(X, Y, params, cache):
    m_batch = 128
    dZ2 = cache["A2"] - Y
    dW2 = (1./m_batch) * np.matmul(dZ2, cache["A1"].T)
    db2 = (1./m_batch) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(params["W2"].T, dZ2)
    dZ1 = dA1 * sigmoid(cache["Z1"]) * (1 - sigmoid(cache["Z1"]))
    dW1 = (1./m_batch) * np.matmul(dZ1, X.T)
    db1 = (1./m_batch) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads

X_train, X_test = X_train.reshape(60000,-1) / 255, X_test.reshape(10000,-1) / 255
X_train, X_test = X_train.T, X_test.T

y_train, y_test = one_hot(y_train), one_hot(y_test)



#A = sigmoid(Z)

if True:
    ################################################## 

    learning_rate = 1

    X = X_train
    Y = y_train
    n_x = X.shape[0]
    m = X.shape[1]

    n_h = 64
    learning_rate = 1

    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(10, n_h)
    b2 = np.zeros((10, 1))

for i in range(2000):

    Z1 = np.matmul(W1,X) + b1
    A1 = sigmoid(Z1)


    Z2 = np.matmul(W2,A1) + b2
    A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)

    cost = compute_loss(Y, A2)

    dZ2 = A2-Y
    dW2 = (1./m) * np.matmul(dZ2, A1.T)
    db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(W2.T, dZ2)
    dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))
    dW1 = (1./m) * np.matmul(dZ1, X.T)
    db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    if (i % 100 == 0):
        print("Epoch", i, "cost: ", cost)

print("Final cost:", cost)
np.save("W1.npy", W1)
np.save("W2.npy", W2)

np.save("b1.npy", b1)
np.save("b2.npy", b2)


###################################################

W1 = np.load("W1.npy")
b1 = np.load("b1.npy")
W2 = np.load("W2.npy")
b2 = np.load("b2.npy")

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