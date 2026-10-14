import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from RuleTree import RuleTreeCluster, RuleTreeClusterClassifier, RuleTreeClusterRegressor


@pytest.mark.parametrize("clus_impurity", ["r2", "bic"])
def test_cluster_fit_predict(blobs, clus_impurity):
    X, y = blobs
    model = RuleTreeCluster(max_depth=3, clus_impurity=clus_impurity, random_state=0)

    labels = model.fit_predict(X)

    assert np.issubdtype(labels.dtype, np.integer)
    assert adjusted_rand_score(y, labels) > 0.9
    assert np.array_equal(model.predict(X), labels)


def test_cluster_labels(blobs):
    X, _ = blobs
    model = RuleTreeCluster(max_depth=3).fit(X)

    node_ids = set(model.get_predicates()) | set(model.get_leaf_nodes())
    assert set(model.label_encoder) == node_ids
    assert model.label_encoder["R"] == 0
    assert sorted(model.label_encoder.values()) == list(range(len(node_ids)))
    assert set(model.predict(X)) == {model.label_encoder[leaf_id] for leaf_id in model.get_leaf_nodes()}


def test_cluster_post_fit_fix_is_idempotent(blobs):
    X, _ = blobs
    model = RuleTreeCluster(max_depth=3).fit(X)
    labels = model.predict(X)

    model._post_fit_fix()

    assert np.array_equal(model.predict(X), labels)


def test_cluster_mixed_features(iris_mixed):
    X, _ = iris_mixed
    labels = RuleTreeCluster(max_depth=2).fit_predict(X)

    assert len(np.unique(labels)) > 1


def test_cluster_invalid_impurity():
    with pytest.raises(ValueError):
        RuleTreeCluster(clus_impurity="unknown")


def test_cluster_classifier(iris):
    X, y = iris
    model = RuleTreeClusterClassifier(max_depth=3).fit(X, y)

    y_pred = model.predict(X)
    proba = model.predict_proba(X)

    assert set(y_pred) <= set(y)
    assert np.mean(y_pred == y) > 0.6
    assert proba.shape == (X.shape[0], len(np.unique(y)))
    assert np.allclose(proba.sum(axis=1), 1)


def test_cluster_regressor(diabetes):
    X, y = diabetes
    model = RuleTreeClusterRegressor(max_depth=3, bic_eps=0.5).fit(X, y)

    y_pred = model.predict(X)

    assert len(model.get_leaf_nodes()) > 1
    assert np.issubdtype(y_pred.dtype, np.floating)
    assert y.min() <= y_pred.min() and y_pred.max() <= y.max()


def test_cluster_update_statistics_keeps_the_labels(blobs):
    X, _ = blobs
    model = RuleTreeCluster(max_depth=3, random_state=0).fit(X)
    labels = model.predict(X)

    model.update_statistics(X[:100])

    assert np.array_equal(model.predict(X), labels)
    assert model.root.samples == 100


def test_cluster_regressor_update_statistics(diabetes):
    X, y = diabetes
    model = RuleTreeClusterRegressor(max_depth=3, bic_eps=0.5, random_state=0).fit(X, y)
    predictions = model.predict(X)

    model.update_statistics(X, y)

    assert np.allclose(model.predict(X), predictions)  # means of the target, as in fit
