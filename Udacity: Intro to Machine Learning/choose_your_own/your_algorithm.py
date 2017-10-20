#!/usr/bin/python

"""
    Shubhodeep Mitra
    20/10/2017

    This is the code to accompany the Lesson 5 (Choose-Your-Own Algorithm) mini-project(Udacity).

    Use a KNN,RandomForestClassifier, adaboost to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""



import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary

## Random Forest Classifier
## clf = RandomForestClassifier(n_estimators=10)

## K-Nearest Neighbors
clf=KNeighborsClassifier(n_neighbors=1)

## AdaBoost Classifier
## clf = AdaBoostClassifier()

ini_t=time()
clf = clf.fit(features_train,labels_train)

train_t= time()
print "training time:",str(train_t - ini_t)

pred=clf.predict(features_test)

pred_t=time()
print "prediction time:",str(pred_t - train_t)

acc=accuracy_score(labels_test,pred)

print "accuracy:",str(acc)






try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
