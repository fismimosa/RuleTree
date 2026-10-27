"""
random_state makes every model reproducible: two fits with the same random_state give the same trees,
whatever the global random state of numpy and of random, and the fits do not change the global state.
"""
import random

import numpy as np
import pytest

from RuleTree import (RuleTreeClassifier, RuleTreeCluster, RuleTreeClusterClassifier, RuleTreeClusterRegressor,
                      RuleTreeRegressor)
from RuleTree.ensemble.EBMRegressor import EBMRegressor
from RuleTree.ensemble.GBoostClassifier import GBoostClassifier
from RuleTree.ensemble.GBoostRegressor import GBoostRegressor
from RuleTree.stumps import classification as clf, regression as reg

from .test_stumps import STUMPS, time_series  # noqa: F401 (fixture of the stumps)


def describe(model):
    """The rule of each internal node and the prediction of each leaf, without rounding."""
    nodes = model.get_predicates() | model.get_leaf_nodes()
    return {node_id: node.stump.get_rule(float_precision=None)["textual_rule"] if not node.is_leaf()
            else repr(np.asarray(node.prediction).tolist()) for node_id, node in nodes.items()}


def fit_twice(make, fit):
    """The results of two fits of new models, after setting the global random state differently."""
    results = []
    for global_seed in (1, 2):
        np.random.seed(global_seed)
        random.seed(global_seed)
        model = make()
        results.append((np.asarray(fit(model)), describe(model) if hasattr(model, "root") else None))
    return results


def assert_reproducible(make, fit):
    (predictions_a, tree_a), (predictions_b, tree_b) = fit_twice(make, fit)
    assert tree_a == tree_b
    np.testing.assert_array_equal(predictions_a, predictions_b)


@pytest.mark.parametrize("make_stump, data_type, dataset, continuous", STUMPS,
                         ids=[f"{s[0]().__class__.__name__}-{s[1]}" for s in STUMPS])
def test_every_stump(make_stump, data_type, dataset, continuous, request):
    X, y = request.getfixturevalue(dataset)
    y = y.astype(float) if continuous else y
    tree = RuleTreeRegressor if continuous else RuleTreeClassifier
    argument = {"tabular": "X", "ts": "X_ts"}[data_type]

    assert_reproducible(lambda: tree(max_depth=3, random_state=0, base_stumps=make_stump()),
                        lambda model: model.fit(**{argument: X}, y=y).predict(**{argument: X}))


def several_stumps():
    return [clf.DecisionTreeStumpClassifier(), clf.PivotTreeStumpClassifier(),
            clf.ObliqueDecisionTreeStumpClassifier(), clf.MultiplePivotTreeStumpClassifier()]


@pytest.mark.parametrize("stump_selection", ["random", "best"])
def test_several_stumps(stump_selection, iris):
    X, y = iris

    assert_reproducible(
        lambda: RuleTreeClassifier(max_depth=4, random_state=0, stump_selection=stump_selection,
                                   base_stumps=several_stumps()),
        lambda model: model.fit(X, y).predict(X))


def test_partial_fit_and_random_state_instance(iris):
    X, y = iris

    assert_reproducible(
        lambda: RuleTreeClassifier(max_depth=2, random_state=0, base_stumps=several_stumps()),
        lambda model: model.fit(X, y).set_params(max_depth=4).partial_fit(X, y).predict(X))
    assert_reproducible(
        lambda: RuleTreeClassifier(max_depth=4, random_state=np.random.RandomState(0), base_stumps=several_stumps()),
        lambda model: model.fit(X, y).predict(X))


def test_without_random_state_the_random_choices_differ(iris):
    """The check above can fail: without random_state, the stumps drawn at random change between fits."""
    X, y = iris

    (_, tree_a), (_, tree_b) = fit_twice(lambda: RuleTreeClassifier(max_depth=4, base_stumps=several_stumps()),
                                         lambda model: model.fit(X, y).predict(X))

    assert tree_a != tree_b


@pytest.mark.parametrize("stump, component_split", [
    (None, "best"), (None, "joint"),
    (reg.ObliqueDecisionTreeStumpRegressor(), "best"), (reg.ObliqueDecisionTreeStumpRegressor(), "joint"),
    (reg.PivotTreeStumpRegressor(), "best"), (reg.PivotTreeStumpRegressor(), "joint"),
], ids=["default-best", "default-joint", "oblique-best", "oblique-joint", "pivot-best", "pivot-joint"])
def test_clustering(stump, component_split, blobs):
    X, _ = blobs

    assert_reproducible(
        lambda: RuleTreeCluster(max_depth=4, random_state=0, component_split=component_split,
                                base_stumps=None if stump is None else type(stump)()),
        lambda model: model.fit_predict(X))


def test_clustering_trees_with_target(iris, diabetes):
    X, y = iris
    X_d, y_d = diabetes

    assert_reproducible(lambda: RuleTreeClusterClassifier(max_depth=4, random_state=0),
                        lambda model: model.fit(X, y).predict(X))
    assert_reproducible(lambda: RuleTreeClusterRegressor(max_depth=4, random_state=0, bic_eps=0.5),
                        lambda model: model.fit(X_d, y_d).predict(X_d))


def random_regression_tree(**kwargs):
    """A base estimator that draws its stump at random, without its own random_state."""
    return RuleTreeRegressor(max_depth=2, base_stumps=[reg.DecisionTreeStumpRegressor(),
                                                       reg.ObliqueDecisionTreeStumpRegressor(),
                                                       reg.PivotTreeStumpRegressor()], **kwargs)


@pytest.mark.parametrize("make, dataset", [
    (lambda: GBoostRegressor(base_estimator=random_regression_tree(), n_estimators=8, random_state=0), "diabetes"),
    (lambda: GBoostClassifier(base_estimator=random_regression_tree(), n_estimators=8, n_jobs=1, random_state=0),
     "iris"),
    (lambda: EBMRegressor(base_estimator=random_regression_tree(), n_iterations=2, random_state=0), "diabetes"),
], ids=["GBoostRegressor", "GBoostClassifier", "EBMRegressor"])
def test_ensembles_seed_the_trees_without_random_state(make, dataset, request):
    X, y = request.getfixturevalue(dataset)

    assert_reproducible(make, lambda model: model.fit(X, y).predict(X))


def test_ensembles_give_each_tree_its_seed_and_keep_the_explicit_one(diabetes):
    X, y = diabetes

    drawn = GBoostRegressor(base_estimator=random_regression_tree(), n_estimators=8, patience=100,
                            random_state=0).fit(X, y)
    explicit = GBoostRegressor(base_estimator=random_regression_tree(random_state=7), n_estimators=8, patience=100,
                               random_state=0).fit(X, y)

    seeds = [tree.random_state for tree in drawn.estimators_]
    assert None not in seeds and len(set(seeds)) == len(seeds)
    assert {tree.random_state for tree in explicit.estimators_} == {7}


@pytest.mark.slow
def test_gboost_classifier_same_result_with_any_number_of_processes(iris):
    X, y = iris
    make = lambda n_jobs: GBoostClassifier(base_estimator=random_regression_tree(), n_estimators=5,  # noqa: E731
                                           n_jobs=n_jobs, random_state=0)

    np.testing.assert_array_equal(make(1).fit(X, y).predict(X), make(2).fit(X, y).predict(X))


@pytest.mark.parametrize("fit", [
    lambda X, y: RuleTreeClassifier(max_depth=3, random_state=0, base_stumps=several_stumps()).fit(X, y),
    lambda X, y: RuleTreeCluster(max_depth=3, random_state=0).fit(X),
    lambda X, y: GBoostRegressor(base_estimator=random_regression_tree(), n_estimators=3).fit(X, y),
    lambda X, y: EBMRegressor(base_estimator=random_regression_tree(), n_iterations=1).fit(X, y),
    lambda X, y: GBoostClassifier(base_estimator=random_regression_tree(), n_estimators=3, n_jobs=1).fit(X, y),
], ids=["RuleTreeClassifier", "RuleTreeCluster", "GBoostRegressor", "EBMRegressor", "GBoostClassifier"])
def test_fit_does_not_change_the_global_random_state(fit, iris):
    X, y = iris
    np.random.seed(123)
    random.seed(123)
    expected = (np.random.random_sample(3), random.random())

    np.random.seed(123)
    random.seed(123)
    fit(X, y)

    np.testing.assert_array_equal(np.random.random_sample(3), expected[0])
    assert random.random() == expected[1]
