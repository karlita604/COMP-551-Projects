import numpy as np
import pandas as pd

class NaiveBayes:
    pass

    def __init__(self):
        pass
    
    def computePriors(self, X, y) :
        classes = np.unique(y)
        priors = np.zeros(len(classes))
        for i, cl in enumerate(classes) :
            occurences = np.count_nonzero(y==cl)
            priors[i] = occurences/X.shape[0]
        return priors

    def calcMeans(self, X, y) :
        # calculate mean and standard deviation
        means = np.zeros(len(np.unique(y)))
        stdev = np.zeros(len(np.unique(y)))
        for i, cl in enumerate(np.unique(y)) :
            arr = X[y==cl]
            means[i] = np.mean(arr)
            stdev[i] = np.std(arr)
        return means, stdev

    def calcProb(self, X, y) :
        prob = np.zeros(len(np.unique(y)))
        for i, cl in enumerate(np.unique(y)) :
            arr = X[y==cl]
            prob[i] = len(arr)/len(y)
        return prob

    def fit(self, X, y, types) : 
        nb_classes = len(np.unique(y))
        nb_features = X.shape[1]
        priors = self.computePriors(X, y)
        means = np.zeros((nb_features, nb_classes))
        stdev = np.zeros((nb_features, nb_classes))
        prob = np.zeros((nb_features, nb_classes))
        for i, feat in enumerate(types) :
            if feat == 0 :
                prob[i,:] = self.calcProb(X[:,i], y)
            elif feat == 1 :
                means[i,:], stdev[i,:] = self.calcMeans(X[:,i], y)
        return priors, means, stdev, prob

    def predict(self, X_test, mu, s, prob, priors, types):
        predictions = np.zeros((X_test.shape[0], 1)) #one prediction for each instance
        for i in range(len(predictions)):
            posteriors = np.zeros((len(priors), 1))
            likelihood = np.zeros((len(priors), 1))
            for j, feat in enumerate(types):
                if feat == 0:
                    #categorical
                    for k, cl in enumerate(priors):
                        likelihood += np.log(prob[j,k])
                else: #continuous
                    for k, cl in enumerate(priors):
                        exponent = np.exp(-((X_test[i,j] - mu[j, k])**2 / (2 * s[j,k]**2)))
                        likelihood[k] += np.log((1/(np.sqrt(2*np.pi) * s[j, k]))*exponent)
            for a, post in enumerate(posteriors) :   
                posteriors[a] = np.log(priors[a]) + likelihood[a]
            predictions[i] = np.argmax(posteriors)
        return predictions

    def eval_acc(self, predictions, y_test):
        count = 0
        for i, y_i in enumerate(y_test):
            if y_test[i] == (int)(predictions[i]):
                count += 1
        return count/len(y_test)