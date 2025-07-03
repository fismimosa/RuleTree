import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    adjusted_rand_index,
)
from RuleTree.tree import (
    RuleTreeClassifier,
    RuleTreeRegressor,
    RuleTreeCluster,
    RuleTreeClusterRegressor,
    RuleTreeClusterClassifier,
)

def test_rule_tree_models(
    dataset_path_clf="datasets/CLF/iris.csv",
    dataset_path_reg="datasets/REG/boston.csv",
    dataset_path_cluster="datasets/CLU/aggregation.csv",
):
    # Funzione per valutare i modelli di classificazione
    def evaluate_classifier(model, X_test, y_test):
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        return {"accuracy": accuracy, "report": report}

    # Funzione per valutare i modelli di regressione
    def evaluate_regressor(model, X_test, y_test):
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        return {"rmse": rmse}

    # Funzione per valutare i modelli di clustering
    def evaluate_clusterer(model, X_test, y_test):
        clusters = model.fit_predict(X_test)
        ari = adjusted_rand_index(y_test, clusters)
        return {"ari": ari}

    # Carica i dataset
    data_clf = pd.read_csv(dataset_path_clf)
    data_reg = pd.read_csv(dataset_path_reg)
    data_cluster = pd.read_csv(dataset_path_cluster)

    # Prepara i dati per la classificazione
    X_clf = data_clf.drop('target', axis=1)
    y_clf = data_clf['target']
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )

    # Prepara i dati per la regressione (usa la penultima feature)
    X_reg = data_reg.iloc[:, :-2]  # Assume che l'ultima feature sia target e la seconda ultima sia da usare
    y_reg = data_reg.iloc[:, -1]
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    # Prepara i dati per il clustering (usa la penultima feature)
    X_cluster = data_cluster.iloc[:, :-1]  # Assume che l'ultima colonna sia target
    y_cluster = data_cluster.iloc[:, -1]
    X_train_cluster, X_test_cluster, y_train_cluster, y_test_cluster = train_test_split(
        X_cluster, y_cluster, test_size=0.2, random_state=42
    )

    # Test RuleTreeClassifier
    print("\nTesting RuleTreeClassifier...")
    clf = RuleTreeClassifier(max_depth=3, min_samples_leaf=5)
    clf.fit(X_train_clf, y_train_clf)
    results_clf = evaluate_classifier(clf, X_test_clf, y_test_clf)
    print(f"Accuracy: {results_clf['accuracy']}")
    print("Classification Report:")
    print(pd.DataFrame(results_clf['report']).transpose())

    # Test RuleTreeRegressor
    print("\nTesting RuleTreeRegressor...")
    reg = RuleTreeRegressor(max_depth=3, min_samples_leaf=5)
    reg.fit(X_train_reg, y_train_reg)
    results_reg = evaluate_regressor(reg, X_test_reg, y_test_reg)
    print(f"RMSE: {results_reg['rmse']}")

    # Test RuleTreeCluster
    print("\nTesting RuleTreeCluster...")
    cluster = RuleTreeCluster(n_clusters=3, max_depth=3, min_samples_leaf=5)
    cluster.fit(X_train_cluster)
    results_cluster = evaluate_clusterer(cluster, X_test_cluster, y_test_cluster)
    print(f"Adjusted Rand Index: {results_cluster['ari']}")

    # Test RuleTreeClusterRegressor
    print("\nTesting RuleTreeClusterRegressor...")
    cluster_reg = RuleTreeClusterRegressor(
        n_clusters=3, max_depth=3, min_samples_leaf=5
    )
    cluster_reg.fit(X_train_reg, y_train_reg)
    results_cluster_reg = evaluate_regressor(cluster_reg, X_test_reg, y_test_reg)
    print(f"RMSE: {results_cluster_reg['rmse']}")

    # Test RuleTreeClusterClassifier
    print("\nTesting RuleTreeClusterClassifier...")
    cluster_clf = RuleTreeClusterClassifier(
        n_clusters=3, max_depth=3, min_samples_leaf=5
    )
    cluster_clf.fit(X_train_clf, y_train_clf)
    results_cluster_clf = evaluate_classifier(cluster_clf, X_test_clf, y_test_clf)
    print(f"Accuracy: {results_cluster_clf['accuracy']}")
    print("Classification Report:")
    print(pd.DataFrame(results_cluster_clf['report']).transpose())

if __name__ == '__main__':
    test_rule_tree_models()