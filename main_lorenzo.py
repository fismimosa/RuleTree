import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

from RuleTree import RuleTreeClassifier
from RuleTree.stumps.classification.matrix.DecisionTreeStumpClassifierMatrixV1 import \
    DecisionTreeStumpClassifierMatrixV1
from RuleTree.stumps.classification.matrix.DecisionTreeStumpClassifierMatrixV2 import \
    DecisionTreeStumpClassifierMatrixV2

if __name__ == "__main__":
    df = pd.read_csv("datasets/CLF/diabetes.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    start = time.time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Time taken to train model: ", time.time() - start)
    print(classification_report(y_test, y_pred))

    clf = RuleTreeClassifier(max_depth=2, random_state=42)
    start = time.time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Time taken to train model: ", time.time() - start)
    print(classification_report(y_test, y_pred))

    clf = RuleTreeClassifier(max_depth=2,
                             base_stumps=DecisionTreeStumpClassifierMatrixV1(random_state=42, batch_size=1),
                             random_state=42)
    start = time.time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("[MATRIX V1] Time taken to train model: ", time.time() - start)
    print(classification_report(y_test, y_pred))

    clf = RuleTreeClassifier(max_depth=2,
                             base_stumps=DecisionTreeStumpClassifierMatrixV2(random_state=42),
                             random_state=42)
    start = time.time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("[MATRIX V2] Time taken to train model: ", time.time() - start)
    print(classification_report(y_test, y_pred))
