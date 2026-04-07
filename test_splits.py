import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingRegressor, BaggingClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from RuleTree import RuleTreeClassifier
import pickle

from RuleTree.stumps.classification import DecisionTreeStumpClassifier


def test_dtc(X_ord, y):
    seed = 0
    while True:
        dt = DecisionTreeClassifier(max_depth=3, random_state=seed).fit(X_ord, y)
        thr = np.unique(dt.tree_.threshold)
        if np.sum((thr != .5) & (thr != -2.)) != 0:
            print(seed, thr)
            break
        if seed % 100 == 0:
            print(seed)
        seed += 1

def test_rtc(X_ord, y):
    seed = 0
    while True:
        dt = RuleTreeClassifier(max_depth=1, random_state=seed).fit(X_ord, y)
        thr = np.unique(dt.root.stump.tree_.threshold)
        if np.sum((thr != .5) & (thr != -2.)) != 0:
            print(seed, thr)
            break
        if seed % 100 == 0:
            print(seed)
        seed += 1

def test_rfc(X_ord, y):
    seed = 0
    while True:
        rf = RandomForestClassifier(n_estimators=1000, max_depth=1, random_state=seed, n_jobs=-1).fit(X_ord, y)
        for dt in rf.estimators_:
            thr = np.unique(dt.tree_.threshold)
            if np.sum((thr != .5) & (thr != -2.)) != 0:
                print(seed, thr)
                return
        if seed % 100 == 0:
            print(seed)
        seed += 1

def test_rufc(X_ord, y):
    seed = 0
    while True:
        rf = BaggingClassifier(n_estimators=1000, estimator=RuleTreeClassifier(max_depth=1),
                                    random_state=seed, n_jobs=-1).fit(X_ord, y)
        for dt in rf.estimators_:
            thr = np.unique(dt.root.stump.tree_.threshold)
            if np.sum((thr != .5) & (thr != -2.)) != 0:
                print(seed, thr)
                return
        if seed % 100 == 0:
            print(seed)
        seed += 1

if __name__ == "__main__":
    X = np.array([0, 1]).reshape(-1, 1)
    y = np.array([0, 1])

    dt = RuleTreeClassifier(max_depth=1, random_state=42,
                            base_stumps=DecisionTreeStumpClassifier(max_depth=1, random_state=42, splitter='random')
                            ).fit(X, y)

    print(dt.root.stump.threshold_original)


