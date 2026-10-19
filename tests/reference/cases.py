"""
Configurations used by the non-regression test (test_reference.py).

Each case fits a model on a fixed data set and returns its outputs (predictions, leaves,
probabilities, rules). The outputs of the current code are stored in predictions.json by
``python -m tests.reference.generate``; the test checks that the code keeps producing them.
Regenerate the file only for intended changes of behavior.
"""
import numpy as np
from sklearn.datasets import load_diabetes, load_iris, make_blobs

from RuleTree import (RuleTreeClassifier, RuleTreeCluster, RuleTreeClusterClassifier, RuleTreeClusterRegressor,
                      RuleTreeRegressor)
from RuleTree.ensemble.EBMRegressor import EBMRegressor
from RuleTree.ensemble.GBoostClassifier import GBoostClassifier
from RuleTree.ensemble.GBoostRegressor import GBoostRegressor
from RuleTree.ensemble.XGBoostRegressor import XGBoostRegressor
from RuleTree.stumps.classification import (DecisionTreeStumpClassifier, MultipleObliquePivotTreeStumpClassifier,
                                            MultiplePivotTreeStumpClassifier, ObliqueDecisionTreeStumpClassifier,
                                            ObliquePivotTreeStumpClassifier, PartialPivotTreeStumpClassifier,
                                            PartialProximityTreeStumpClassifier, PivotTreeStumpClassifier,
                                            ProximityTreeStumpClassifier, ShapeletTreeStumpClassifier)
from RuleTree.stumps.regression import (DecisionTreeStumpRegressor, FairTreeStumpRegressor,
                                        ObliqueDecisionTreeStumpRegressor, PivotTreeStumpRegressor,
                                        ProximityTreeStumpRegressor, ShapeletTreeStumpRegressor)
from RuleTree.stumps.regression.XGBoostStumpRegressor import XGBoostStumpRegressor
from RuleTree.tree.RuleTree import RuleTree
from RuleTree.tree.RuleTreeXGBoostRegressor import RuleTreeXGBoostRegressor


def _datasets():
    X_iris, y_iris = load_iris(return_X_y=True)
    X_diabetes, y_diabetes = load_diabetes(return_X_y=True)
    X_blobs, y_blobs = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=0.5, random_state=0)

    color = np.where(X_iris[:, 2] < 2.5, "red", np.where(X_iris[:, 2] < 4.8, "green", "blue"))
    X_mixed = np.empty((len(y_iris), 3), dtype=object)  # column_stack would turn the numbers into strings
    X_mixed[:, 0], X_mixed[:, 1:] = color, X_iris[:, :2]

    rng = np.random.default_rng(0)
    y_ts = rng.integers(0, 2, size=80)
    X_ts = rng.normal(scale=0.3, size=(80, 1, 40))
    X_ts[y_ts == 1, 0, 15:25] += 2

    X_iris_ts = rng.normal(scale=0.3, size=(len(y_iris), 1, 30))
    X_iris_ts[y_iris == 2, 0, 10:20] += 2

    return {
        "iris": {"X": X_iris},
        "iris_y": y_iris,
        "mixed": {"X": X_mixed},
        "mixed_y_reg": X_iris[:, 2],
        "diabetes": {"X": X_diabetes},
        "diabetes_y": y_diabetes,
        "diabetes_Y": np.column_stack([y_diabetes, X_diabetes[:, 2]]),  # two targets on different scales
        "fair": {"X": np.column_stack([X_diabetes, X_diabetes[:, 1] > 0])},
        "blobs": {"X": X_blobs},
        "ts": {"X_ts": X_ts},
        "ts_y": y_ts,
        "multimodal": {"X": X_iris[:, :2], "X_ts": X_iris_ts},
    }


DATA = _datasets()


# Stumps backed by a sklearn tree get a fixed random_state: with random_state=None, sklearn breaks
# ties between features of equal gain at random, and the outputs would change between runs.
def dt_clf(**kwargs):
    return DecisionTreeStumpClassifier(random_state=0, **kwargs)


def dt_reg(**kwargs):
    return DecisionTreeStumpRegressor(random_state=0, **kwargs)


def pivot_clf(**kwargs):
    return PivotTreeStumpClassifier(random_state=0, **kwargs)


def pivot_reg(**kwargs):
    return PivotTreeStumpRegressor(random_state=0, **kwargs)


def partial_pivot_clf(**kwargs):
    return PartialPivotTreeStumpClassifier(random_state=0, **kwargs)


def shapelet_clf():
    return ShapeletTreeStumpClassifier(n_shapelets=4, sliding_window=10, selection="random")


def shapelet_reg():
    return ShapeletTreeStumpRegressor(n_shapelets=4, sliding_window=10, selection="random", mi_n_neighbors=5)


# name -> (model factory, input data, target, rules are compared)
# Shapelet rules are not compared: get_rule draws the shapelet into a temporary file.
CASES = {
    # classification
    "clf_default": (lambda: RuleTreeClassifier(max_depth=4, base_stumps=dt_clf()), "iris", "iris_y", True),
    "clf_entropy": (lambda: RuleTreeClassifier(max_depth=4, base_stumps=dt_clf(criterion="entropy")),
                    "iris", "iris_y", True),
    "clf_min_samples_leaf": (lambda: RuleTreeClassifier(max_depth=4,
                                                        base_stumps=dt_clf(min_samples_leaf=10)),
                             "iris", "iris_y", True),
    "clf_max_leaf_nodes": (lambda: RuleTreeClassifier(max_leaf_nodes=5, base_stumps=dt_clf()), "iris", "iris_y", True),
    "clf_prune": (lambda: RuleTreeClassifier(max_depth=6, prune_useless_leaves=True, base_stumps=dt_clf()), "iris", "iris_y", True),
    "clf_categorical_best": (lambda: RuleTreeClassifier(max_depth=4, stump_selection="best", base_stumps=dt_clf()), "mixed", "iris_y", True),
    "clf_pivot": (lambda: RuleTreeClassifier(max_depth=3, base_stumps=pivot_clf()), "iris", "iris_y", True),
    "clf_pivot_cosine": (lambda: RuleTreeClassifier(max_depth=3,
                                                    base_stumps=pivot_clf(distance_measure="cosine")),
                         "iris", "iris_y", True),
    "clf_partial_pivot": (lambda: RuleTreeClassifier(max_depth=2, base_stumps=partial_pivot_clf()),
                          "iris", "iris_y", True),
    "clf_best_mixed_stumps": (lambda: RuleTreeClassifier(max_depth=3, stump_selection="best",
                                                         base_stumps=[dt_clf(),
                                                                      pivot_clf()]),
                              "iris", "iris_y", True),
    "clf_oblique_householder": (lambda: RuleTreeClassifier(
        max_depth=3, base_stumps=ObliqueDecisionTreeStumpClassifier(random_state=0)), "iris", "iris_y", True),
    "clf_oblique_bivariate": (lambda: RuleTreeClassifier(
        max_depth=3, base_stumps=ObliqueDecisionTreeStumpClassifier("bivariate", random_state=0)),
                              "iris", "iris_y", True),
    "clf_multiple_pivot": (lambda: RuleTreeClassifier(
        max_depth=3, base_stumps=MultiplePivotTreeStumpClassifier(random_state=0)), "iris", "iris_y", True),
    "clf_oblique_pivot": (lambda: RuleTreeClassifier(
        max_depth=3, base_stumps=ObliquePivotTreeStumpClassifier(random_state=0)), "iris", "iris_y", True),
    "clf_multiple_oblique_pivot": (lambda: RuleTreeClassifier(
        max_depth=3, base_stumps=MultipleObliquePivotTreeStumpClassifier(oblique_split_type="bivariate",
                                                                         random_state=0)), "iris", "iris_y", True),
    "clf_partial_proximity": (lambda: RuleTreeClassifier(
        max_depth=2, base_stumps=PartialProximityTreeStumpClassifier(random_state=0)), "iris", "iris_y", True),
    "clf_proximity": (lambda: RuleTreeClassifier(
        max_depth=2, base_stumps=ProximityTreeStumpClassifier(n_shapelets=4, sliding_window=10, selection="random")),
                      "ts", "ts_y", True),
    "clf_decision_tree_ts": (lambda: RuleTreeClassifier(max_depth=3, base_stumps=dt_clf()), "ts", "ts_y", True),
    "clf_shapelet": (lambda: RuleTreeClassifier(max_depth=2, base_stumps=shapelet_clf()), "ts", "ts_y", False),
    "clf_multimodal_best": (lambda: RuleTreeClassifier(max_depth=3, stump_selection="best",
                                                       base_stumps=[dt_clf(), shapelet_clf()]),
                            "multimodal", "iris_y", False),
    "clf_multimodal_random": (lambda: RuleTreeClassifier(max_depth=3, random_state=0,
                                                         base_stumps=[(0.5, dt_clf(), "tabular"),
                                                                      (0.5, shapelet_clf())]),
                              "multimodal", "iris_y", False),
    # regression
    "reg_default": (lambda: RuleTreeRegressor(max_depth=4, base_stumps=dt_reg()), "diabetes", "diabetes_y", True),
    "reg_absolute_error": (lambda: RuleTreeRegressor(max_depth=3,
                                                     base_stumps=dt_reg(criterion="absolute_error")),
                           "diabetes", "diabetes_y", True),
    "reg_categorical": (lambda: RuleTreeRegressor(max_depth=3, base_stumps=dt_reg()), "mixed", "mixed_y_reg", True),
    "reg_pivot": (lambda: RuleTreeRegressor(max_depth=2, base_stumps=pivot_reg()),
                  "diabetes", "diabetes_y", True),
    "reg_best_mixed_stumps": (lambda: RuleTreeRegressor(max_depth=2, stump_selection="best",
                                                        base_stumps=[dt_reg(),
                                                                     pivot_reg()]),
                              "diabetes", "diabetes_y", True),
    "reg_shapelet": (lambda: RuleTreeRegressor(max_depth=2, base_stumps=shapelet_reg()), "ts", "ts_y", False),
    "reg_oblique": (lambda: RuleTreeRegressor(max_depth=2,
                                              base_stumps=ObliqueDecisionTreeStumpRegressor(random_state=0)),
                    "diabetes", "diabetes_y", True),
    "reg_proximity": (lambda: RuleTreeRegressor(max_depth=2, base_stumps=ProximityTreeStumpRegressor(
        n_shapelets=4, sliding_window=10, selection="random", mi_n_neighbors=5)), "ts", "ts_y", True),
    "reg_multi_target": (lambda: RuleTreeRegressor(max_depth=3, base_stumps=dt_reg()), "diabetes", "diabetes_Y", True),
    "reg_multi_target_balanced": (lambda: RuleTreeRegressor(max_depth=3, base_stumps=dt_reg(),
                                                            target_weights="balanced", leaf_prediction="median"),
                                  "diabetes", "diabetes_Y", True),
    "reg_xgboost_stump": (lambda: RuleTreeRegressor(max_depth=3, base_stumps=XGBoostStumpRegressor(lam=1)),
                          "diabetes", "diabetes_y", True),
    "reg_fair": (lambda: RuleTreeRegressor(max_depth=2, base_stumps=FairTreeStumpRegressor(
        penalty="balance", sensible_attribute=-1, n_jobs=1)), "fair", "diabetes_y", True),
    "reg_xgboost_tree": (lambda: RuleTreeXGBoostRegressor(max_depth=3, base_stumps=XGBoostStumpRegressor(lam=1)),
                         "diabetes", "diabetes_y", True),
    # clustering
    "clu_r2": (lambda: RuleTreeCluster(max_depth=3, random_state=0, base_stumps=dt_reg()), "blobs", None, True),
    "clu_bic": (lambda: RuleTreeCluster(max_depth=3, clus_impurity="bic", random_state=0, base_stumps=dt_reg()), "blobs", None, True),
    "clu_oblique": (lambda: RuleTreeCluster(max_depth=3, random_state=0,
                                            base_stumps=ObliqueDecisionTreeStumpRegressor("bivariate")), "blobs", None, True),
    "clu_mixed": (lambda: RuleTreeCluster(max_depth=2, random_state=0, base_stumps=dt_reg()), "mixed", None, True),
    "clu_classifier": (lambda: RuleTreeClusterClassifier(max_depth=3, random_state=0, base_stumps=dt_reg()), "iris", "iris_y", True),
    "clu_regressor": (lambda: RuleTreeClusterRegressor(max_depth=3, bic_eps=0.5, random_state=0, base_stumps=dt_reg()),
                      "diabetes", "diabetes_y", True),
    # ensembles
    "ens_gboost_regressor": (lambda: GBoostRegressor(base_estimator=RuleTreeRegressor(max_depth=2, base_stumps=dt_reg()), n_estimators=5),
                             "diabetes", "diabetes_y", False),
    "ens_gboost_classifier": (lambda: GBoostClassifier(base_estimator=RuleTreeRegressor(max_depth=2, base_stumps=dt_reg()), n_estimators=5),
                              "iris", "iris_y", False),
    "ens_xgboost_regressor": (lambda: XGBoostRegressor(n_estimators=3), "diabetes", "diabetes_y", False),
    "ens_ebm_regressor": (lambda: EBMRegressor(base_estimator=RuleTreeRegressor(max_depth=2, base_stumps=dt_reg()), n_iterations=2,
                                               random_state=0), "diabetes", "diabetes_y", False),
}

# grown in two steps with partial_fit
PARTIAL_FIT_CASE = "clf_partial_fit"


def _rules(model):
    rules = {}
    for node_id, node in model.get_predicates().items():
        rules[node_id] = node.stump.get_rule()["textual_rule"]
    return rules


def outputs(model, inputs: dict, rules: bool) -> dict:
    """Outputs of a fitted model on its training inputs, as JSON-friendly lists."""
    result = {"predict": np.asarray(model.predict(**inputs)).tolist()}

    if isinstance(model, RuleTree):
        result["apply"] = np.asarray(model.apply(**inputs)).tolist()
        if isinstance(model, RuleTreeClassifier) or isinstance(model, RuleTreeClusterClassifier):
            result["predict_proba"] = np.asarray(model.predict_proba(**inputs)).tolist()
        if rules:
            result["rules"] = _rules(model)

    return result


def fit_case(name: str):
    """Fit the model of a case; returns (model, inputs, outputs)."""
    if name == PARTIAL_FIT_CASE:
        inputs = DATA["iris"]
        model = RuleTreeClassifier(max_depth=1, base_stumps=dt_clf()).fit(y=DATA["iris_y"], **inputs)
        model.set_params(max_depth=3).partial_fit(y=DATA["iris_y"], **inputs)
        return model, inputs, outputs(model, inputs, rules=True)

    factory, data_name, target_name, rules = CASES[name]
    inputs = DATA[data_name]
    target = DATA[target_name] if target_name is not None else None
    model = factory()

    if isinstance(model, RuleTree):
        model.fit(y=target, **inputs)
    else:  # ensembles take only tabular data
        model.fit(inputs["X"], target)

    return model, inputs, outputs(model, inputs, rules)


ALL_CASES = list(CASES) + [PARTIAL_FIT_CASE]
