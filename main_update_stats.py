import copy

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from RuleTree import RuleTreeClassifier


def test_update_statistics():
    df = pd.read_csv("datasets/CLF/iris.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    encoder = LabelEncoder()
    y_train_new = encoder.fit_transform(y_train)
    y_test_new = encoder.transform(y_test)

    params = dict(max_depth=6, stump_selection="best", random_state=42)

    tree_old = RuleTreeClassifier(**params)
    tree_old.fit(X_train, y_train)

    tree_retrained = RuleTreeClassifier(**params)
    tree_retrained.fit(X_train, y_train_new)

    tree_updated = copy.deepcopy(tree_old)
    tree_updated.update_statistics(X_train, y_train_new)

    pred_retrained = tree_retrained.predict(X_test)
    pred_updated = tree_updated.predict(X_test)

    acc_retrained = accuracy_score(y_test_new, pred_retrained)
    acc_updated = accuracy_score(y_test_new, pred_updated)
    f1_retrained = f1_score(y_test_new, pred_retrained, average="weighted")
    f1_updated = f1_score(y_test_new, pred_updated, average="weighted")

    print(f"Accuracy retrained: {acc_retrained:.6f}")
    print(f"Accuracy updated  : {acc_updated:.6f}")
    print(f"F1 retrained      : {f1_retrained:.6f}")
    print(f"F1 updated        : {f1_updated:.6f}")

    assert np.isclose(acc_retrained, acc_updated)
    assert np.isclose(f1_retrained, f1_updated)

    print("Ok")


if __name__ == "__main__":
    test_update_statistics()
