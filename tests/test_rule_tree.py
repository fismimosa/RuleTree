import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, adjusted_rand_score

from RuleTree import RuleTreeClassifier, RuleTreeRegressor, RuleTreeCluster, RuleTreeClusterRegressor, RuleTreeClusterClassifier


def evaluate_classifier(model, X_test, y_test):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        return {"accuracy": accuracy, "report": report}


def evaluate_regressor(model, X_test, y_test):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        return {"rmse": rmse}


def evaluate_cluster(model, X_test, y_test):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        clusters = model.fit_predict(X_test)
        ari = adjusted_rand_score(y_test, clusters)
        return {"ari": ari}


def load_and_split_data(dataset_path):
    data = pd.read_csv(dataset_path)
    return data


def prepare_classification_data(data):
    X = data.drop(columns=['class']).to_numpy()
    y = data['class'].to_numpy()
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_regression_data(data):
    X = data.iloc[:, :-2].to_numpy()
    y = data.iloc[:, -2].to_numpy()
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_clustering_data(data):
    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_rule_tree_classifier(data_path):
    data = load_and_split_data(data_path)
    X_train, X_test, y_train, y_test = prepare_classification_data(data)
    print("\nTesting RuleTreeClassifier...")
    clf = RuleTreeClassifier(max_depth=2)
    clf.fit(X_train, y_train)
    res = evaluate_classifier(clf, X_test, y_test)
    print(f"Accuracy: {res['accuracy']}")
    print("Classification Report:")
    print(pd.DataFrame(res['report']).transpose())


def test_rule_tree_regressor(data_path):
    data = load_and_split_data(data_path)
    X_train, X_test, y_train, y_test = prepare_regression_data(data)
    print("\nTesting RuleTreeRegressor...")
    reg = RuleTreeRegressor(max_depth=3)
    reg.fit(X_train, y_train)
    res = evaluate_regressor(reg, X_test, y_test)
    print(f"RMSE: {res['rmse']}")


def test_rule_tree_cluster(data_path):
    data = load_and_split_data(data_path)
    X_train, X_test, y_train, y_test = prepare_clustering_data(data)
    print("\nTesting RuleTreeCluster...")
    cluster = RuleTreeCluster(max_depth=3)
    cluster.fit(X_train)
    res = evaluate_cluster(cluster, X_test, y_test)
    print(f"Adjusted Rand Index: {res['ari']}")


def test_rule_tree_cluster_regressor(data_path):
    data = load_and_split_data(data_path)
    X_train, X_test, y_train, y_test = prepare_regression_data(data)
    print("\nTesting RuleTreeClusterRegressor...")
    model = RuleTreeClusterRegressor(max_depth=3)
    model.fit(X_train, y_train)
    res = evaluate_regressor(model, X_test, y_test)
    print(f"RMSE: {res['rmse']}")


def test_rule_tree_cluster_classifier(data_path):
    data = load_and_split_data(data_path)
    X_train, X_test, y_train, y_test = prepare_classification_data(data)
    print("\nTesting RuleTreeClusterClassifier...")
    model = RuleTreeClusterClassifier(max_depth=3)
    model.fit(X_train, y_train)
    res = evaluate_classifier(model, X_test, y_test)
    print(f"Accuracy: {res['accuracy']}")
    print("Classification Report:")
    print(pd.DataFrame(res['report']).transpose())


def test_rule_tree_models(dataset_path_clf, dataset_path_reg, dataset_path_cluster):
    test_rule_tree_classifier(dataset_path_clf)
    test_rule_tree_regressor(dataset_path_reg)
    test_rule_tree_cluster(dataset_path_cluster)
    test_rule_tree_cluster_regressor(dataset_path_reg)
    test_rule_tree_cluster_classifier(dataset_path_clf)


if __name__ == '__main__':
    dataset_path_clf="../datasets/CLF/iris.csv"
    dataset_path_reg="../datasets/CLF/iris.csv"
    dataset_path_cluster="../datasets/CLF/iris.csv"

    test_rule_tree_classifier(dataset_path_clf)
    test_rule_tree_regressor(dataset_path_reg)
    test_rule_tree_cluster(dataset_path_cluster)
    test_rule_tree_cluster_regressor(dataset_path_reg)
    test_rule_tree_cluster_classifier(dataset_path_clf)
