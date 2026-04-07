import copy

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from RuleTree import RuleTreeClassifier


def test_update_statistics():
    df = pd.read_csv("datasets/CLF/iris.csv")
    X = df.iloc[:, :-1].values
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    params = dict(max_depth=6, stump_selection="best", random_state=42)

    dt = RuleTreeClassifier(**params)
    dt.fit(X_train, y_train)

    print(classification_report(dt.predict(X_test), y_test))

    dt.update_statistics(X_train, y_train*10)

    print(classification_report(dt.predict(X_test), y_test*10))

    print("Ok")


if __name__ == "__main__":
    test_update_statistics()
