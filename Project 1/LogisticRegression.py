import numpy as np
import pandas as pd

class LogisticRegression:

    def __init__(self, lr, nbIter):
        self.lr = lr
        self.nbIter = nbIter

    def sigmoid(self, x) :
        return 1/(1+ np.exp(-x))

    def cost(self,w,X,y) :
        #w = weights (1xN)
        #X = N training data with d features each
        #y = classification of each training data point
        z = np.dot(X,w)
        J = np.mean(y * np.log1p(np.exp(-z)) + (1-y) * np.log1p(np.exp(z)))
        return J

    #this function outputs a vector of weights
    def fit(self, X, y) :
        #init with random weights
        W = np.random.rand(X.shape[1])
        b = 0.1 #bias
        for i in range(self.nbIter):
            z = np.dot(W.T, X.T) + b
            y_pred = self.sigmoid(z)
            #gradient descent to update weights
            W -= self.lr* self.gradientDescent(X, y, y_pred)
        return W

    def gradientDescent(self, X, y, y_pred) :
        #print(X.shape)
        #print(y_pred.shape)
        #print(y.shape)
        X2 = X.T
        y_pred2 = y_pred.T
        return np.dot(X2, (y_pred2 - y)) / y.shape[0]

    def predict(self, X_test, W):
        result = self.sigmoid(np.dot(X_test, W))
        return np.rint(result) 

    def eval_acc(self, predictions, y_test):
        count = 0
        for i, y_i in enumerate(y_test):
            if y_test[i] == predictions[i]:
                count += 1
        return count/len(y_test)



