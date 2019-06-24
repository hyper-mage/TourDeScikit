from sklearn import datasets

iris = datasets.load_iris()
digits = datasets.load_digits()

print(digits.data)
print(digits.target)

# SVM estimator
from sklearn import svm

clf = svm.SVC(gamma=0.001, C=100)

# Typically we would want to use Grid Search or Cross Validation to tune our gamma variable
# [:-1] produces an array that contains all but the last item from the attached data

clf.fit(digits.data[:-1], digits.target[:-1])
# svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr',
#         degree=3, gamma=0.001, kernel='rbf', max_iter=-1, probability=False, random_state=None,
#         shrinking=True, tol=0.001, verbose=False)
print(clf.predict(digits.data[-1:]))

# next lets work on some model persistence, or mnist data.

# We proceed with model persistence, lets finish this tutorial and move on to personal projects afterwards

# Model Persistence
# we utilize python's built in serializer, pickle. Pickle basically bytes and unbytes objects

clf = svm.SVC(gamma='scale')
iris = datasets.load_iris()
X, y = iris.data, iris.target  # that is super dope that i can assign variables like this
clf.fit(X, y)
# svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr',
#         degree=3, gamma='scale', kernel='rbf', max_iter=-1, probability=False, random_state=None,
#         shrinking=True, tol=0.001, verbose=False)

import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
print(clf2.predict(X[0:1]))
print(y[0])

# Use joblib, a "big data" version of pickle

from joblib import load, dump
dump(clf, 'filename.joblib')

clf = load('filename.joblib')

# The goal of joblib is to save models for later use

