import numpy as np
import pytest

from RuleTree import RuleTreeClassifier
from RuleTree.stumps.classification import PivotTreeStumpClassifier
from RuleTree.utils.tree_utils import _eval_rule, get_feature_node_matrix, get_leaf_internal_node_matrix, \
    get_leaf_prediction_matrix, get_thresholds_matrix


@pytest.mark.parametrize("max_depth", [2, 4])
def test_matrices_predict_as_the_tree(iris, max_depth):
    X, y = iris
    model = RuleTreeClassifier(max_depth=max_depth, random_state=0).fit(X, y)
    predicates, leaves = model.get_predicates(), model.get_leaf_nodes()

    A = get_feature_node_matrix(predicates, X.shape[1])
    B = get_thresholds_matrix(predicates)
    C, internal_nodes = get_leaf_internal_node_matrix(leaves)
    D = np.clip(C, 0, None).sum(axis=0)
    E = get_leaf_prediction_matrix(leaves)

    assert internal_nodes == list(predicates)
    assert A.shape == (X.shape[1], len(predicates)) and A.sum(axis=0).tolist() == [1] * len(predicates)
    predictions = model.classes_[np.argmax((((X @ A <= B) @ C) == D) @ E, axis=1)]
    assert np.array_equal(predictions, model.predict(X))


def test_matrices_need_thresholds_on_columns(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=1, base_stumps=PivotTreeStumpClassifier(), random_state=0).fit(X, y)

    with pytest.raises(ValueError, match="not a threshold"):
        get_feature_node_matrix(model.get_predicates(), X.shape[1])
    with pytest.raises(ValueError, match="not a threshold"):
        get_thresholds_matrix(model.get_predicates())


def test_eval_rule(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=2, random_state=0).fit(X, y)

    labels = _eval_rule(X, model.root)

    assert np.array_equal(labels, np.where(X[:, 3] <= model.root.stump.threshold_original, 1, 2))
    with pytest.raises(ValueError):
        _eval_rule(X, model.get_node_by_id("Rl"))


def test_stumps_to_trees(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=3, random_state=0).fit(X, y)

    trees = model.stumps_to_trees(model.get_balanced_stumps(p=0.1))

    assert len(trees) == len(model.get_balanced_stumps(p=0.1)) > 1
    for (feature, threshold), tree in trees.items():
        assert list(tree.get_leaf_nodes()) == ["Rl", "Rr"]
        assert tree.root.parent is None
        expected = np.where(X[:, feature] <= threshold, tree.root.node_l.prediction, tree.root.node_r.prediction)
        assert np.array_equal(tree.predict(X), expected)
    assert len(model.get_leaf_nodes()) > 2  # the original tree is unchanged
