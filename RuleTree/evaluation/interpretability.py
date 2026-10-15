"""Size of a tree, as a measure of how easy it is to read."""

from sklearn.tree import BaseDecisionTree

from RuleTree.tree.RuleTree import RuleTree


def evaluate_interpretability(model) -> dict:
    """
    Size of a fitted RuleTree or scikit-learn decision tree: the fewer leaves and the shallower
    the tree, the shorter its rules.

    Parameters
    ----------
    model : RuleTree or sklearn.tree.BaseDecisionTree
        Fitted tree.

    Returns
    -------
    scores : dict
        ``n_leaves`` (number of rules), ``n_internal_nodes`` (number of splits) and ``max_depth``
        (length of the longest rule; the root has depth 0). Empty for other models.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from RuleTree import RuleTreeClassifier
    >>> from RuleTree.evaluation import evaluate_interpretability
    >>> X, y = load_iris(return_X_y=True)
    >>> evaluate_interpretability(RuleTreeClassifier(max_depth=3, random_state=0).fit(X, y))
    {'n_leaves': 5, 'n_internal_nodes': 4, 'max_depth': 3}
    """
    if isinstance(model, RuleTree):
        leaves = model.get_leaf_nodes()
        return {
            "n_leaves": len(leaves),
            "n_internal_nodes": len(model.get_predicates()),
            "max_depth": max(leaf.get_depth() for leaf in leaves.values()),
        }
    if isinstance(model, BaseDecisionTree):
        return {
            "n_leaves": int(model.tree_.n_leaves),
            "n_internal_nodes": int(model.tree_.node_count - model.tree_.n_leaves),
            "max_depth": int(model.tree_.max_depth),
        }
    return {}
