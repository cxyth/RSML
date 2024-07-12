# --------------------------------------------------------
# Author     : Jiang
# Email      : cxyth@live.com
# Description: None
# --------------------------------------------------------
import sys
import math
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class SVM_RBF():
    def __init__(self, data, label):
        self.name = 'SVM_RBF'
        self.trainx = data
        self.trainy = label

    def train(self):
        cost = []
        gamma = []
        for i in range(-3, 10, 2):
            cost.append(np.power(2.0, i))
        for i in range(-5, 4, 2):
            gamma.append(np.power(2.0, i))

        parameters = {'C': cost, 'gamma': gamma}
        svm = SVC(verbose=0, kernel='rbf')
        clf = GridSearchCV(svm, parameters, cv=3)
        clf.fit(self.trainx, self.trainy)

        # print(clf.best_params_)
        bestc = clf.best_params_['C']
        bestg = clf.best_params_['gamma']
        tmpc = [-1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0.0,
                0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        cost = []
        gamma = []
        for i in tmpc:
            cost.append(bestc * np.power(2.0, i))
            gamma.append(bestg * np.power(2.0, i))
        parameters = {'C': cost, 'gamma': gamma}
        svm = SVC(verbose=0, kernel='rbf')
        clf = GridSearchCV(svm, parameters, cv=3)
        clf.fit(self.trainx, self.trainy)
        print('best params:', clf.best_params_)
        p = clf.best_estimator_
        return p


class DecisionTree():
    def __init__(self, data, label):
        self.name = 'DecisionTree'
        self.trainx = data
        self.trainy = label

    def train(self):
        model_DD = DecisionTreeClassifier()
        max_depth = range(1, 10, 1)
        min_samples_leaf = range(1, 10, 2)
        tuned_parameters = dict(max_depth=max_depth, min_samples_leaf=min_samples_leaf)

        DD = GridSearchCV(model_DD, tuned_parameters, cv=10)
        DD.fit(self.trainx, self.trainy)
        p = DD.best_estimator_
        return p


class RandomForest():
    def __init__(self, data, label):
        self.name = 'RandomForest'
        self.trainx = data
        self.trainy = label

    def train(self):
        parameter_space = {
            "n_estimators": [20, 50, 100],  # [10, 15, 20],
            "criterion": ["gini", "entropy"],
            "min_samples_leaf": [2, 4, 6],
        }
        scores = ['precision_micro']
        #scores = ['precision_macro']
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()
            clf = RandomForestClassifier(random_state=14)
            grid = GridSearchCV(clf, parameter_space, cv=5, scoring='%s' % score)
            # scoring='%s_macro' % score：precision_macro、recall_macro是用于multiclass/multilabel任务的
            grid.fit(self.trainx, self.trainy)
        p = grid
        return p
