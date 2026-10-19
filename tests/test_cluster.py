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


@pytest.fixture(scope="module")
def square():
    """Four groups at the corners of a square."""
    rng = np.random.default_rng(0)
    corners = np.array([[0, 0], [4, 0], [0, 4], [4, 4]])
    y = rng.integers(0, 4, size=400)
    return corners[y] + rng.normal(size=(400, 2)), y


def test_joint_component_split_is_the_multi_target_split_of_the_components(square):
    """The root split predicts all the principal components at once."""
    from sklearn.tree import DecisionTreeRegressor

    from RuleTree.utils import light_famd

    X, _ = square
    model = RuleTreeCluster(max_depth=1, random_state=0, component_split="joint").fit(X)

    components = light_famd.PCA(n_components=2, random_state=0).fit_transform(X)
    reference = DecisionTreeRegressor(max_depth=1, random_state=0).fit(X, components)
    assert model.root.stump.feature_original == reference.tree_.feature[0]
    assert np.isclose(model.root.stump.threshold_original, reference.tree_.threshold[0])


def test_joint_component_split_on_the_square(square):
    X, y = square
    labels = {component_split: RuleTreeCluster(max_depth=4, random_state=0, component_split=component_split)
              .fit_predict(X) for component_split in ("best", "joint")}

    assert adjusted_rand_score(y, labels["joint"]) > 0.85
    assert adjusted_rand_score(y, labels["joint"]) > adjusted_rand_score(y, labels["best"])


def test_joint_component_split_needs_stumps_with_several_targets(blobs):
    from RuleTree.stumps.regression import FairTreeStumpRegressor

    X, _ = blobs
    data = np.column_stack([X, X[:, 0] > 0]).astype(float)
    model = RuleTreeCluster(max_depth=2, random_state=0, component_split="joint",
                            base_stumps=FairTreeStumpRegressor(penalty="balance", sensible_attribute=2, n_jobs=1))

    with pytest.raises(ValueError, match="supports a single target"):
        model.fit(data)


def test_joint_component_split_in_all_the_cluster_trees(iris, diabetes):
    import json

    from RuleTree.tree.RuleTree import RuleTree
    from RuleTree.utils.data_utils import json_numpy_encoder

    X, y = iris
    classifier = RuleTreeClusterClassifier(max_depth=3, random_state=0, component_split="joint").fit(X, y)
    X_reg, y_reg = diabetes
    regressor = RuleTreeClusterRegressor(max_depth=2, random_state=0, component_split="joint").fit(X_reg, y_reg)

    assert classifier.score(X, y) > 0.6  # two clusters, as with component_split="best": setosa and the rest
    assert regressor.predict(X_reg).shape == y_reg.shape
    loaded = RuleTree.from_dict(json.loads(json.dumps(classifier.to_dict(), cls=json_numpy_encoder)))
    assert loaded.component_split == "joint"
    assert np.array_equal(loaded.predict(X), classifier.predict(X))


def test_invalid_component_split():
    with pytest.raises(ValueError, match="component_split"):
        RuleTreeCluster(component_split="all")
