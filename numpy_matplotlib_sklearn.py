import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata

# download and read mnist
mnist = fetch_mldata('MNIST original')

# 'mnist.data' is 70k x 784 array, each row represents the pixels from a 28x28=784 image
# 'mnist.target' is 70k x 1 array, each row represents the target class of the corresponding image
images = mnist.data
targets = mnist.target

# make the value of pixels from [0, 255] to [0, 1] for further process
X = mnist.data / 255.
Y = mnist.target

# split data to train and test (for faster calculation, just use 1/10 data)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)

# TODO:use logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

lr = LogisticRegression()
lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)
train_accuracy = lr.score(X_train, Y_train)
test_accuracy = metrics.accuracy_score(Y_test,Y_pred)

print('\n',"=======use logistic regression=======")
print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))

# TODO:use naive bayes
from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB()
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
train_accuracy = clf.score(X_train, Y_train)
test_accuracy = metrics.accuracy_score(Y_test,Y_pred)

print('\n',"=======use naive bayes=======")
print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))

# TODO:use support vector machine
from sklearn.svm import LinearSVC

svc = LinearSVC()
svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)
train_accuracy = svc.score(X_train, Y_train)
test_accuracy = metrics.accuracy_score(Y_test,Y_pred)

print('\n',"=======use SVM=======")
print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))

# TODO:use SVM with another group of parameters
from sklearn.svm import LinearSVC

svc = LinearSVC(dual=False,tol=0.000005, C=0.05, class_weight='balanced')
svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)
train_accuracy = svc.score(X_train, Y_train)
test_accuracy = metrics.accuracy_score(Y_test,Y_pred)
print('\n',"=======use SVM after adjust dual, class_weight, C, tol=======")
print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))