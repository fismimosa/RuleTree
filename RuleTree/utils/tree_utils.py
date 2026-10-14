"""Matrix representations of a fitted RuleTree, and the similarity and cover functions of XGBoost."""

import numpy as np

from RuleTree.stumps.classification import DecisionTreeStumpClassifier
from RuleTree.stumps.regression import DecisionTreeStumpRegressor
from RuleTree.tree.RuleTreeNode import RuleTreeNode


def get_nodes_in_path(leaf_node_id: str):
    """
    Ids of the internal nodes on the path from the root to a node.

    Parameters
    ----------
    leaf_node_id : str
        Id of the node, e.g. ``"Rlr"``.

    Returns
    -------
    path : list of str
        Ids from the root to the parent of the node, e.g. ``["R", "Rl"]``.
    """
    return [leaf_node_id[:i] for i in range(1, len(leaf_node_id))]


def get_feature_node_matrix(rules: dict[str, RuleTreeNode]) -> np.ndarray:
    """
    Matrix of the features used by the splits: entry (feature, rule) is 1 if the rule uses the feature.

    Parameters
    ----------
    rules : dict
        Node id -> internal RuleTreeNode, e.g. RuleTree.get_predicates().

    Returns
    -------
    feature_matrix : numpy.ndarray of shape (n_features, n_rules)

    Notes
    -----
    Reads the number of features from the attribute ``n_features`` of the nodes, which the trees do
    not set: it must be set on the nodes before calling this function.
    """
    num_rules = len(rules)
    num_features = list(rules.values())[0].n_features
    feature_matrix = np.zeros((num_features, num_rules), dtype=int)

    for idx_rule, (rule_id, rule) in enumerate(rules.items()):
        for feature_idx in rule.stump.feature_original:
            if feature_idx >= 0:
                feature_matrix[feature_idx, idx_rule] = 1

    return feature_matrix


def get_thresholds_matrix(rules: dict[str, RuleTreeNode]) -> np.ndarray:
    """
    Threshold of each split, for univariate decision tree stumps.

    Parameters
    ----------
    rules : dict
        Node id -> internal RuleTreeNode with a DecisionTreeStumpClassifier or
        DecisionTreeStumpRegressor, e.g. RuleTree.get_predicates().

    Returns
    -------
    thresholds : numpy.ndarray of shape (n_rules,)

    Raises
    ------
    ValueError
        If a node has another kind of stump.

    Notes
    -----
    As get_feature_node_matrix, requires the attribute ``n_features`` on the nodes.
    """
    num_rules = len(rules)
    num_features = list(rules.values())[0].n_features
    thresholds_matrix = np.zeros((num_features, num_rules), dtype=float)

    for idx_rule, (rule_id, rule) in enumerate(rules.items()):
        if type(rule.stump) not in [DecisionTreeStumpClassifier, DecisionTreeStumpRegressor]:
            raise ValueError(f"Unsupported rule stump type: {type(rule)}")

        for feature_idx, threshold in zip(rule.stump.feature_original, rule.stump.threshold_original):
            if feature_idx >= 0:
                thresholds_matrix[feature_idx, idx_rule] = threshold

    return np.sum(thresholds_matrix, axis=0)


def get_leaf_internal_node_matrix(leaf_nodes: dict[str, RuleTreeNode]) -> [np.ndarray, list]:
    """
    Position of each leaf with respect to each internal node on its path: entry (node, leaf) is 1 if
    the leaf is in the left subtree of the node, -1 if in the right subtree, 0 otherwise.

    Parameters
    ----------
    leaf_nodes : dict
        Leaf id -> RuleTreeNode, e.g. RuleTree.get_leaf_nodes().

    Returns
    -------
    matrix : numpy.ndarray of shape (n_internal_nodes, n_leaves)
    internal_nodes : list of str
        Ids of the internal nodes, in the order of the rows.
    """
    internal_nodes = {}
    for leaf_node_id in leaf_nodes.keys():
        for node_in_path in get_nodes_in_path(leaf_node_id):
            internal_nodes[node_in_path] = 1
    internal_nodes = list(internal_nodes.keys())

    leaf_node_matrix = np.zeros((len(internal_nodes), len(leaf_nodes)), dtype=int)

    for leaf_node_idx, leaf_node_id in enumerate(leaf_nodes.keys()):
        for internal_node_idx, internal_node_id in enumerate(internal_nodes):
            if leaf_node_id.startswith(internal_node_id):
                position = leaf_node_id[len(internal_node_id)]
                leaf_node_matrix[internal_node_idx, leaf_node_idx] = 1 if position == 'l' else -1

    return leaf_node_matrix, internal_nodes

def get_leaf_prediction_matrix(leaf_nodes: dict[str, RuleTreeNode], return_proba=False) -> np.ndarray:
    """
    Prediction of each leaf of a classification tree: one-hot encoded class or class probabilities.

    Parameters
    ----------
    leaf_nodes : dict
        Leaf id -> RuleTreeNode, e.g. RuleTree.get_leaf_nodes().
    return_proba : bool, default=False
        Whether to return the class probabilities instead of the one-hot encoded class.

    Returns
    -------
    predictions : numpy.ndarray of shape (n_leaves, n_classes)
    """
    n_classes = len(list(leaf_nodes.values())[0].classes)
    leaf_prediction_matrix = np.zeros((len(leaf_nodes), n_classes))

    for leaf_node_idx, (leaf_node_id, leaf_node) in enumerate(leaf_nodes.items()):
        if return_proba:
            leaf_prediction_matrix[leaf_node_idx] = leaf_node.prediction_probability
        else:
            leaf_prediction_matrix[leaf_node_idx, list(leaf_node.classes).index(leaf_node.prediction)] = 1

    return leaf_prediction_matrix

def _eval_rule(X: np.ndarray, rule: RuleTreeNode, return_probababilities=False):
    """
    Evaluate the split of an internal node.

    Parameters
    ----------
    X : numpy.ndarray
        Data.
    rule : RuleTreeNode
        Internal node.
    return_probababilities : bool, default=False
        Not used.

    Raises
    ------
    ValueError
        If the node is a leaf.

    Notes
    -----
    Calls ``stump.predict``, which the stumps do not implement (they implement ``apply``).
    """
    if rule.is_leaf():
        raise ValueError("The rule is a leaf node, cannot evaluate.")

    rule.stump.predict(X)

def xgboost_similarity_score_regression(residuals:np.ndarray, cover:int, _lambda:int=0) -> float:
    """
    XGBoost similarity score of a node, for regression.

    Parameters
    ----------
    residuals : numpy.ndarray
        Residuals of the instances in the node.
    cover : int
        Cover of the node (see xgboost_cover_regression).
    _lambda : float, default=0
        L2 regularization.

    Returns
    -------
    similarity : float
        ``sum(residuals) ** 2 / (cover + lambda)``.
    """
    return np.square(np.sum(residuals))/(_lambda+cover)

def xgboost_cover_regression(residuals:np.ndarray) -> float:
    """
    XGBoost cover of a node, for regression: the number of instances.

    Parameters
    ----------
    residuals : numpy.ndarray
        Residuals of the instances in the node.

    Returns
    -------
    cover : int
    """
    return len(residuals)

def xgboost_similarity_score_classification(residuals:np.ndarray, cover:float, _lambda:int=0) -> float:
    """
    XGBoost similarity score of a node, for classification.

    Parameters
    ----------
    residuals : numpy.ndarray
        Residuals of the instances in the node.
    cover : float
        Cover of the node (see xgboost_cover_classification).
    _lambda : float, default=0
        L2 regularization.

    Returns
    -------
    similarity : float
        ``sum(residuals) ** 2 / (cover + lambda)``.
    """
    return np.square(np.sum(residuals))/(cover+_lambda)

def xgboost_cover_classification(prev_probability:np.ndarray) -> float:
    """
    XGBoost cover of a node, for classification.

    Parameters
    ----------
    prev_probability : numpy.ndarray
        Probabilities predicted for the instances in the node.

    Returns
    -------
    cover : float
        ``sum(p * (1 - p))``.
    """
    return np.sum(prev_probability*(1-prev_probability))