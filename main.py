import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

from benchmark.competitors.kmeanstree import KMeansTree
from RuleTree import RuleTreeClassifier, RuleTreeClusterClassifier, RuleTreeRegressor


def test_clf_iris():
    df = pd.read_csv("datasets/CLF/iris.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf_rule = RuleTreeClassifier(max_depth=5)

    clf_rule.fit(X_train, y_train)

    f1_rule = f1_score(y_test, clf_rule.predict(X_test), average='weighted')

    print(f"F1: {f1_rule}")

    print(clf_rule.print_rules(clf_rule.get_rules(columns_names=df.columns)))

def test_reg_iris():
    df = pd.read_csv("datasets/CLF/iris.csv")
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
    X_train = df_train.iloc[:, :-2].values
    X_test = df_test.iloc[:, :-2].values
    y_train = df_train.iloc[:, -2].values
    y_test = df_test.iloc[:, -2].values

    reg_rule = RuleTreeRegressor(max_depth=5)

    reg_rule.fit(X_train, y_train)

    f1_rule = mean_squared_error(y_test, reg_rule.predict(X_test))

    print(f"MSE: {f1_rule}")


def test_clc_home():
    df = pd.read_csv("datasets/CLF/home.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf_rule = RuleTreeClusterClassifier()

    clf_rule.fit(X_train, y_train)

    f1_rule = f1_score(y_test, clf_rule.predict(X_test), average='weighted')

    print(f"F1: {f1_rule}")


def test_clf_iris_incremental():
    df = pd.read_csv("datasets/CLF/iris.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf_rule = RuleTreeClassifier(max_depth=1)

    clf_rule.fit(X_train, y_train)
    f1_rule = f1_score(y_test, clf_rule.predict(X_test), average='weighted')
    print(f"F1: {f1_rule}")
    clf_rule.print_rules(clf_rule.get_rules(columns_names=df.columns))

    clf_rule.max_depth = 2
    clf_rule.fit(X_train, y_train)
    f1_rule = f1_score(y_test, clf_rule.predict(X_test), average='weighted')
    print(f"F1: {f1_rule}")
    clf_rule.print_rules(clf_rule.get_rules(columns_names=df.columns))

    clf_rule.max_depth = 3
    clf_rule.fit(X_train, y_train)
    f1_rule = f1_score(y_test, clf_rule.predict(X_test), average='weighted')
    print(f"F1: {f1_rule}")
    clf_rule.print_rules(clf_rule.get_rules(columns_names=df.columns))

def test_reg_iris_incremental():
    df = pd.read_csv("datasets/CLF/iris.csv")
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
    X_train = df_train.iloc[:, :-2].values
    X_test = df_test.iloc[:, :-2].values
    y_train = df_train.iloc[:, -2].values
    y_test = df_test.iloc[:, -2].values

    reg_rule = RuleTreeRegressor(max_depth=1)

    reg_rule.fit(X_train, y_train)
    mse = mean_squared_error(y_test, reg_rule.predict(X_test))
    print(f"MSE: {mse}")
    reg_rule.print_rules(reg_rule.get_rules(columns_names=df.columns))

    reg_rule.max_depth = 2
    reg_rule.fit(X_train, y_train)
    mse = mean_squared_error(y_test, reg_rule.predict(X_test))
    print(f"MSE: {mse}")
    reg_rule.print_rules(reg_rule.get_rules(columns_names=df.columns))

    reg_rule.max_depth = 3
    reg_rule.fit(X_train, y_train)
    mse = mean_squared_error(y_test, reg_rule.predict(X_test))
    print(f"MSE: {mse}")
    reg_rule.print_rules(reg_rule.get_rules(columns_names=df.columns))




if __name__ == "__main__":
    #test_clf_iris()
    #test_clc_home()
    #test_reg_iris()
    #test_clf_iris_incremental()
    test_reg_iris_incremental()
