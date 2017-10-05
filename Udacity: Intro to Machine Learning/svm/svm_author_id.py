#!/usr/bin/python

"""
    Shubhodeep Mitra
    05/10/2017

    This is the code to accompany the Lesson 2 (SVM) mini-project(Udacity).

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################

#clf=SVC(kernel='linear') #using linear kernel mode in place of default rbf mode

clf=SVC(kernel='rbf',C=10000)

'''
#slice the training dataset down to 1% of its original size, tossing out 99% of the training data
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]
'''

in_time=time()   #taking note of intial time

clf.fit(features_train,labels_train)
train_time=time()   #taking note of the time after training phase

print "training time is",(train_time- in_time) #time taken to train in seconds

pred=clf.predict(features_test)
pred_time=time() #time at which prediction is complete

print "prediction time is",(pred_time-train_time) #time taken to complete prediction

acc=accuracy_score(labels_test,pred)

print acc


#task: to print pred[i], where i=10,26,50
print "10:",pred[10]," 26:",pred[26]," 50:",pred[50]

#task: to count how many emails were from chris(label 1)

ch_count=0
for i in pred:
    if i==1:
        ch_count+=1

print "No of chris email:%d"%(ch_count)
