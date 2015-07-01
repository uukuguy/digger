#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''

classifier.py

'''

from __future__ import division
import logging

from sklearn import svm
from sklearn import metrics
from sklearn.metrics import accuracy_score


# ================ class Classifier ================
class Classifier():

    # ---------------- __init__() ----------------
    def __init__(self):
        self.clf = svm.LinearSVC()
        #self.clf = svm.SVC(kernel='rbf', C=1.0)

    # ---------------- print_predict_result() ----------------
    def print_predict_result(self, clf, y_pred, y_test, categories_names):
        clf_descr = str(clf).split('(')[0]
        score = accuracy_score(y_test, y_pred)
        print
        print "<<", clf_descr, ">>"
        #print "Training time: %.3f" % (train_time)
        #print "Testing time: %.3f" % (test_time)
        print
        print "Accuracy: %.3f" % (score)
        print "Classification Report:"
        print metrics.classification_report(y_test, y_pred, target_names=categories_names)
        print "Confusion Matrix:"
        print metrics.confusion_matrix(y_test, y_pred)

    # ---------------- train() ----------------
    def train(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    # ---------------- predict() ----------------
    def predict(self, X_test, y_test, categories_names):
        y_pred = self.clf.predict(X_test)

        self.print_predict_result(self.clf, y_pred, y_test, categories_names)

        return y_pred



