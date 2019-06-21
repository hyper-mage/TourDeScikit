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
svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr',
        degree=3, gamma=0.001, kernel='rbf', max_iter=-1, probability=False, random_state=None,
        shrinking=True, tol=0.001, verbose=False)
print(clf.predict(digits.data[-1:]))

# next lets work on some model persistence, or mnist data.
