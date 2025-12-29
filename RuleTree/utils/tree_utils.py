import numpy as np
from sklearn.tree import DecisionTreeRegressor

from RuleTree.stumps.classification import DecisionTreeStumpClassifier
from RuleTree.stumps.regression import DecisionTreeStumpRegressor
from RuleTree.tree.RuleTreeNode import RuleTreeNode


def get_nodes_in_path(leaf_node_id: str):
    """
    Given the id of a leaf node, return the list of all nodes in the path from the root to the leaf node.

    :param leaf_node_id: The id of the leaf node.
    :return: A list of node ids in the path from the root to the leaf node.
    """
    return [leaf_node_id[:i] for i in range(1, len(leaf_node_id))]


def get_feature_node_matrix(rules: dict[str, RuleTreeNode]) -> np.ndarray:
    """
    Given a list of rules, return a matrix where each row represents a rule and each column represents a feature.
    The value in the matrix is 1 if the feature is used in the rule, otherwise 0.

    :param rules: A dictionary of rules.
    :return: A matrix representing the features used in the rules.
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
    Given a list of rules, return a matrix where each row represents a rule and each column represents a feature.
    The value in the matrix is the threshold used in the rule.

    :param rules: A dictionary of rules.
    :return: A matrix representing the thresholds used in the rules.
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
    Generate a matrix representing the relationship between internal nodes and leaf nodes in a tree.

    Each row in the matrix corresponds to an internal node, and each column corresponds to a leaf node.
    The value in the matrix is determined as follows:
    - `1` if the corresponding leaf node in the column lies in the left subtree of the corresponding internal node in the row.
    - `-1` if the corresponding leaf node in the column lies in the right subtree of the corresponding internal node in the row.
    - `0` if the leaf node is not in the subtree of the internal node.

    The determination of left or right subtree is based on the last character of the `internal_node_id`:
    - `'l'` indicates a left child.
    - `'r'` indicates a right child. In this implementation, any other character is considered `'r'`.

    Args:
        leaf_nodes (dict[str, RuleTreeNode]): A dictionary where keys are leaf node IDs and values are `RuleTreeNode` objects.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: A 2D matrix of shape (number of internal nodes, number of leaf nodes).
            - list: A list of internal node IDs.
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
    n_classes = len(list(leaf_nodes.values())[0].classes)
    leaf_prediction_matrix = np.zeros((len(leaf_nodes), n_classes))

    for leaf_node_idx, (leaf_node_id, leaf_node) in enumerate(leaf_nodes.items()):
        if return_proba:
            leaf_prediction_matrix[leaf_node_idx] = leaf_node.prediction_probability
        else:
            leaf_prediction_matrix[leaf_node_idx, list(leaf_node.classes).index(leaf_node.prediction)] = 1

    return leaf_prediction_matrix

def _eval_rule(X: np.ndarray, rule: RuleTreeNode, return_probababilities=False):
    if rule.is_leaf():
        raise ValueError("The rule is a leaf node, cannot evaluate.")

    rule.stump.predict(X)

def xgboost_similarity_score(residuals:np.ndarray, _lambda:int=0) -> float:
    return np.square(np.sum(residuals))/(_lambda+len(residuals))