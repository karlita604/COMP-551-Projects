import numpy as np 
import pandas as pd
from time import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn import metrics     #for performance analysis
from sklearn.model_selection import GridSearchCV
import warnings

class FitAndPredictNews:

    def __init__(self, train, test):    #train and test are data frames with the data being x and the target being y
        self.train = train
        self.test = test

    def __run__(self):

        ################################################################################################################

        #---- DECISION TREES -----#
        #param options
        t0 = time()
        criterion = ['gini', 'entropy']
        splitter = ['best', 'random']
        
        param_grid = dict(clf__criterion=criterion, clf__splitter=splitter)

        #Model Pipeline
        dt = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', DecisionTreeClassifier())
        ])

        #fit our classifier on the train data
        print("\n\n\n# DECISION TREES: Tuning hyper-parameters for accuracy #\n")
        dt_clf = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1)
        dt_clf.fit(self.train.data, self.train.target)
        #print out best params
        print("Best parameters set found on development set:")
        print(dt_clf.best_params_)

        #print grid scores
        print("\nGrid scores on development set:")
        means = dt_clf.cv_results_['mean_test_score']
        stds = dt_clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds,dt_clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))


        #Classification report - predict on test set
        print("\nDetailed classification report:")
        y_true, y_pred = self.test.target, dt_clf.predict(self.test.data)
        print(metrics.classification_report(y_true, y_pred, target_names=self.test.target_names))

        print("done in %fs" % (time() - t0))


        ################################################################################################################

        #---- LINEAR SVC ----#
        #param options
        t0 = time()
        loss = ['hinge', 'squared_hinge']
        C = [0.1, 1, 10, 100, 1000]
        max_iter = [50, 100, 150, 200]

        param_grid = dict(clf__loss=loss, clf__C=C, clf__max_iter=max_iter)

        #Model Pipeline
        svc = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', LinearSVC(penalty='l2'))
        ])

        #fit our classifier on the train data
        print("\n\n\n# LINEAR SVC: Tuning hyper-parameters for accuracy #\n")
        svc_clf = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, n_jobs=-1)
        svc_clf.fit(self.train.data, self.train.target)
        #print out best params
        print("Best parameters set found on development set:")
        print(svc_clf.best_params_)

        #print grid scores
        print("\nGrid scores on development set:")
        means = svc_clf.cv_results_['mean_test_score']
        stds = svc_clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, svc_clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))


        #Classification report - predict on test set
        print("\nDetailed classification report:")
        y_true, y_pred = self.test.target, svc_clf.predict(self.test.data)
        print(metrics.classification_report(y_true, y_pred, target_names=self.test.target_names))

        print("done in %fs" % (time() - t0))


        ################################################################################################################

        #---- ADABOOST ----#
        #param options
        t0 = time()
        base_estimator = [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)]
        n_estimators = [10, 25, 50, 75]

        param_grid = dict(clf__base_estimator=base_estimator, clf__n_estimators=n_estimators)

        #Model Pipeline
        ada = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', AdaBoostClassifier())
        ])

        #fit our classifier on the train data
        print("\n\n\n# ADABOOST: Tuning hyper-parameters for accuracy #\n")
        ada_clf = GridSearchCV(estimator=ada, param_grid=param_grid, cv=5, n_jobs=-1)
        ada_clf.fit(self.train.data, self.train.target)
        #print out best params
        print("Best parameters set found on development set:")
        print(ada_clf.best_params_)

        #print grid scores
        print("\nGrid scores on development set:")
        means = ada_clf.cv_results_['mean_test_score']
        stds = ada_clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, ada_clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))


        #Classification report - predict on test set
        print("\nDetailed classification report:")
        y_true, y_pred = self.test.target, ada_clf.predict(self.test.data)
        print(metrics.classification_report(y_true, y_pred, target_names=self.test.target_names))

        print("done in %fs" % (time() - t0))


        ################################################################################################################

        #---- RANDOM FORESTS ----#
        #param options
        t0 = time()
        n_estimators = [75, 100, 150, 200]
        criterion = ['gini', 'entropy']

        param_grid = dict(clf__n_estimators=n_estimators, clf__criterion=criterion)

        #Model Pipeline
        rf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', RandomForestClassifier())
        ])

        #fit our classifier on the train data
        print("\n\n\n# RANDOM FORESTS: Tuning hyper-parameters for accuracy #\n")
        rf_clf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
        rf_clf.fit(self.train.data, self.train.target)
        #print out best params
        print("Best parameters set found on development set:")
        print(rf_clf.best_params_)

        #print grid scores
        print("\nGrid scores on development set:")
        means = rf_clf.cv_results_['mean_test_score']
        stds = rf_clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, rf_clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))


        #Classification report - predict on test set
        print("\nDetailed classification report:")
        y_true, y_pred = self.test.target, rf_clf.predict(self.test.data)
        print(metrics.classification_report(y_true, y_pred, target_names=self.test.target_names))

        print("done in %fs" % (time() - t0))


        ################################################################################################################

        #---- LOGISTIC REGRESSION ----#
        #param options
        t0 = time()
        C = [1, 10, 100, 1000, 1500]
        solver = ['liblinear', 'sag', 'saga']

        param_grid = dict(clf__C=C, clf__solver=solver)

        #Model Pipeline
        lr = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression(n_jobs=-1, penalty='l2', max_iter=100, class_weight='balanced'))
        ])

        #fit our classifier on the train data
        print("\n\n\n# LOGISTIC REGRESSION: Tuning hyper-parameters for accuracy #\n")
        log_clf = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5)
        log_clf.fit(self.train.data, self.train.target)
        #print out best params
        print("Best parameters set found on development set:")
        print(log_clf.best_params_)

        #print grid scores
        print("\nGrid scores on development set:")
        means = log_clf.cv_results_['mean_test_score']
        stds = log_clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, log_clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))


        #Classification report - predict on test set
        print("\nDetailed classification report:")
        y_true, y_pred = self.test.target, log_clf.predict(self.test.data)
        print(metrics.classification_report(y_true, y_pred, target_names=self.test.target_names))

        print("done in %fs" % (time() - t0))


        ################################################################################################################

        #---- NAIVE BAYES ----#
        t0 = time()
        #param options
        use_idf = [True, False]
        norm = ('l1', 'l2')
        alpha = [1, 0.1, 0.01, 0.001]

        param_grid = dict(tfidf__use_idf=use_idf, tfidf__norm=norm, clf__alpha=alpha) 

        #Model Pipeline
        nb = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB())
        ])

        #fit our classifier on the train data
        print("\n\n\n# NAIVE BAYES: Tuning hyper-parameters for accuracy #\n")
        nb_clf = GridSearchCV(estimator=nb, param_grid=param_grid, n_jobs=-1, cv=5)
        nb_clf.fit(self.train.data, self.train.target)
        #print out best params
        print("Best parameters set found on development set:")
        print(nb_clf.best_params_)

        #print grid scores
        print("\nGrid scores on development set:")
        means = nb_clf.cv_results_['mean_test_score']
        stds = nb_clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, nb_clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))


        #Classification report - predict on test set
        print("\nDetailed classification report:")
        y_true, y_pred = self.test.target, nb_clf.predict(self.test.data)
        print(metrics.classification_report(y_true, y_pred, target_names=self.test.target_names))

        print("done in %fs" % (time() - t0))

        ################################################################################################################

        #---- SVM and PERCEPTRON ----#
        t0 = time()
        #param options
        loss = ['hinge', 'perceptron']
        alpha = [0.01, 0.001, 0.0001, 0.00001]


        param_grid = dict(clf__loss=loss, clf__alpha=alpha)

        #Model Pipeline
        svm = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(penalty='l2'))
        ])

        #fit our classifier on the train data
        print("\n\n\n# SVM/PERCEPTRON: Tuning hyper-parameters for accuracy #\n")
        svm_clf = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, n_jobs=-1)
        svm_clf.fit(self.train.data, self.train.target)
        #print out best params
        print("Best parameters set found on development set:")
        print(svm_clf.best_params_)

        #print grid scores
        print("\nGrid scores on development set:")
        means = svm_clf.cv_results_['mean_test_score']
        stds = svm_clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, svm_clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))


        #Classification report - predict on test set
        print("\nDetailed classification report:")
        y_true, y_pred = self.test.target, svm_clf.predict(self.test.data)
        print(metrics.classification_report(y_true, y_pred, target_names=self.test.target_names))

        print("done in %fs" % (time() - t0))


        ################################################################################################################

