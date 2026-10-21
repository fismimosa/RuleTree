"""Matrix representations of a fitted RuleTree, and the similarity and cover functions of XGBoost."""

import numpy as np

from RuleTree.stumps.transformers import MatrixTransformer
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


def get_feature_node_matrix(rules: dict[str, RuleTreeNode], n_features: int) -> np.ndarray:
    """
    Feature of each split, as a matrix: entry (feature, rule) is 1 if the rule splits on the feature.

    Together with :func:`get_thresholds_matrix`, the splits of all the rules are evaluated at once:
    ``X @ A <= B`` is True where an instance satisfies a rule.

    Parameters
    ----------
    rules : dict
        Node id -> internal RuleTreeNode, e.g. ``RuleTree.get_predicates()``. The splits must be
        thresholds on columns of tabular data.
    n_features : int
        Number of features of the data.

    Returns
    -------
    feature_matrix : numpy.ndarray of shape (n_features, n_rules)

    Raises
    ------
    ValueError
        If a split is not a threshold on a column of the data.
    """
    feature_matrix = np.zeros((n_features, len(rules)), dtype=int)
    for idx_rule, rule in enumerate(rules.values()):
        feature_matrix[_split_column(rule), idx_rule] = 1

    return feature_matrix


def get_thresholds_matrix(rules: dict[str, RuleTreeNode]) -> np.ndarray:
    """
    Threshold of each split (see :func:`get_feature_node_matrix`).

    Parameters
    ----------
    rules : dict
        Node id -> internal RuleTreeNode, e.g. ``RuleTree.get_predicates()``. The splits must be
        thresholds on columns of tabular data.

    Returns
    -------
    thresholds : numpy.ndarray of shape (n_rules,)

    Raises
    ------
    ValueError
        If a split is not a threshold on a column of the data.
    """
    for rule in rules.values():
        _split_column(rule)

    return np.array([rule.stump.threshold_original for rule in rules.values()], dtype=float)


def _split_column(rule: RuleTreeNode) -> int:
    """Column of the data split by a rule, checking that the split is a threshold on that column."""
    feature = rule.stump.feature_original
    if not isinstance(rule.stump.transformer_, MatrixTransformer) or rule.stump.is_categorical:
        raise ValueError(f"The split of node {rule.node_id} is not a threshold on a column of the data.")
    return int(feature)


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

def _eval_rule(X: np.ndarray, rule: RuleTreeNode) -> np.ndarray:
    """
    Apply the split of an internal node to the instances of X.

    Parameters
    ----------
    X : numpy.ndarray
        Data of the type used by the split.
    rule : RuleTreeNode
        Internal node.

    Returns
    -------
    labels : numpy.ndarray of shape (n_samples,)
        1 for the instances that go to the left child, 2 for the ones that go to the right child.

    Raises
    ------
    ValueError
        If the node is a leaf.
    """
    if rule.is_leaf():
        raise ValueError("The rule is a leaf node, cannot evaluate.")

    return rule.stump.apply(X, node_id=rule.node_id)


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