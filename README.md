# COMP-551-Projects
Where I store the different Machine Learning Projects developed individually and within teams during Winter 2020

### Project 1:

Grade: 90%

This project had the objective of analyzing four different data sets and implementing Logistic Regression and Naive Bayes models on each of the data sets. The goal was to explore classification and compare different features and models, using 5-fold cross validation in all of the experiments and evaluating the performance using accuracy. We ran the following experiments:
1. Compared the accuracy of Naive Bayes and Logistic Regression on the four datasets.
We found that the logistic regression approach was achieved worse accuracy than naive Bayes and was significantly slower to train.
2. Tested different learning rates for gradient descent applied to logistic regression. We found that there was a ”sweet spot” for the learning rate (around 0.01).
3. Compared the accuracy of the two models as a function of the size of dataset (by controlling the training size).
Naive Bayes seemed to be less affected by the change in test size, whereas Logistic Regression showed a steep decline in accuracy (to that of a complete guess) as the test size increased toward 90%.
4. Results demonstrating that the feature subset used improves performance. Ran on modified Adult Data Set. Most importantly, we found that the ’Sex’ feature for the Adult data set contributed to the classification greatly.

### Project 2:

Grade: 88%

In this project, our task was to categorize text data from two different datasets. The first dataset, which we will refer to as IMDB, consists of media reviews to be classified as positive or negative. The second dataset, which we will refer to as Newsgroups, consists of newsgroups posts to be classified according to which of 20 newsgroups it was posted in. To do this, we used various models from the scikit-learn library and used grid search with cross validation to tune the hyper-parameters. With this approach, we obtained the highest accuracy (88%) on the IMDB dataset using SVM, Logistic Regression, and Linear SVC. On the Newsgroups dataset, we obtained the highest accuracy (70%) using SVM, Naive Bayes, and Linear SVC.

### Project 3:

Grade: 92%

In this project, we investigated the performance of two neural networks - a multi-layer perceptron (MLP) and a convolutional neural network (CNN) - on an image classification task using the CIFAR-10 dataset. The MLP model achieved 55% accuracy, and the CNN achieved 71% accuracy. We found that the number of epochs a model is run for and the learning rate had large impacts on the accuracy of these models.
