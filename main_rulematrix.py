import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from RuleTree import RuleTreeClassifier
from RuleTree.utils.tree_utils import get_feature_node_matrix, get_thresholds_matrix, get_leaf_internal_node_matrix, \
    get_leaf_prediction_matrix

if __name__ == "__main__":
    df = pd.read_csv("datasets/CLF/iris.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    rt = RuleTreeClassifier(
        max_depth=2,
        random_state=42,
    )

    rt.fit(X_train, y_train)

    y_pred = rt.predict(X_test)

    print(classification_report(y_test, y_pred))

    predicates = rt.get_predicates()
    leaf_nodes = rt.get_leaf_nodes()

    rt.export_graphviz()

    A = get_feature_node_matrix(predicates)
    B = get_thresholds_matrix(predicates)
    C, _ = get_leaf_internal_node_matrix(leaf_nodes)
    D = np.copy(C)
    D[D < 0] = 0
    D = np.sum(D, axis=0)
    E = get_leaf_prediction_matrix(leaf_nodes, return_proba=True)

    y_pred_matrix = (((X_test @ A < B) @ C) == D) @ E

    y_pred_matrix = rt.classes_[np.argmax(y_pred_matrix, axis=1)]

    print(classification_report(y_test, y_pred_matrix))
