import pytest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from RuleTree import RuleTreeClassifier


def test_classifier_iris():
    iris = load_iris()
    X = iris.data
    y = np.array(list(map(lambda x: iris.target_names[x], iris.target)))

    y = np.array([f"cl_{_y}" for _y in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)

    max_nbr_values_cat = 4
    rt = RuleTreeClassifier(
        # max_depth=4,
        # min_samples_leaf=1,
        # min_samples_split=2,
        max_nbr_nodes=7,
        # clf_impurity='entropy',
        # feature_names=['Sex', 'Lies', 'Cookies'],
        # feature_names=iris.feature_names,
        # allow_oblique_splits=True,
        # force_oblique_splits=True,
        # max_nbr_values=4,
        # max_nbr_nodes=32,
        prune_useless_leaves=True,
        # max_nbr_nodes=5,
        bic_eps=0.5,
        # max_nbr_values=20,
        max_nbr_values_cat=max_nbr_values_cat,
        one_hot_encode_cat=True,
        # categorical_indices=[4],
        # numerical_indices=[1],
        numerical_scaler=StandardScaler(),
        # exclude_split_feature_from_reduction=True,
        random_state=0,
        verbose=True
    )

    rt.fit(X_train, y_train)

    y_pred, leaves, rules = rt.predict(X_test, get_leaf=True, get_rule=True)

    assert accuracy_score(y_test, y_pred) > .9

if __name__ == "__main__":
    pytest.main()