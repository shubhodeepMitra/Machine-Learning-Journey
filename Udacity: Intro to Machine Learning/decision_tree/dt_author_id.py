#!/usr/bin/python

"""
    Shubhodeep Mitra
    15/10/2017

    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###



t_ini=time()
clf=DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train,labels_train)

t_train=time()
print "time taken to train:",str(t_train-t_ini)
pred=clf.predict(features_test)

t_pred=time()
print "time taken to predict:",str(t_pred-t_train)

acc=accuracy_score(labels_test,pred)
print "accuracy is:",str(acc)


''' Task: To print the number of features(columns) in the training set
'''

print 'Number of features:',len(features_train[0]),


#########################################################
