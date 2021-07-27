import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from adaboost import AdaBoost

data = load_breast_cancer()
X = data.data
Y = data.target
n_classifiers = 400
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print (X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

adaboost_cls = AdaBoost(n_classifiers, len(X_train))
adaboost_cls.fit(X_train, Y_train)
preds = adaboost_cls(X_test)
acc = accuracy_score(Y_test, preds)
print (acc)

plt.plot(range(n_classifiers), adaboost_cls.errors, color="red")
plt.xlabel("stumps")
plt.ylabel("training error")
plt.show()