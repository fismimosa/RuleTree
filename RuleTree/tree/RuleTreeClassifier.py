"""RuleTreeClassifier: the RuleTree for classification."""

from typing import Union, Optional

import numpy as np
from sklearn.base import ClassifierMixin
import copy

from RuleTree.utils.define import MODEL_TYPE_CLF
from RuleTree.tree.RuleTree import RuleTree
from RuleTree.tree.RuleTreeNode import RuleTreeNode
from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from RuleTree.utils.data_utils import calculate_mode, log_odds


class RuleTreeClassifier(ClassifierMixin, RuleTree):
    """
    RuleTree for classification.

    The tree grows breadth-first; a node is not split if it is pure. Each node predicts the majority
    class of its instances and stores the class probabilities and log-odds.

    Parameters
    ----------
    max_leaf_nodes : int, default=inf
        Maximum number of leaves.
    min_samples_split : int, default=2
        Minimum number of instances to split a node.
    max_depth : int, default=inf
        Maximum depth of the tree (the root has depth 0).
    prune_useless_leaves : bool, default=True
        Whether to merge, after fit, the sibling leaves with the same prediction.
    base_stumps : RuleTreeBaseStump or list, default=None
        Stumps used to split the nodes (see :class:`RuleTree.tree.RuleTree.RuleTree`); they must be
        classification stumps. If None, DecisionTreeStumpClassifier().
    stump_selection : {'random', 'best'}, default='random'
        How the stump of each node is chosen among base_stumps.
    random_state : int, RandomState instance or None, default=None
        Seed of the random choice of the stumps, also given to the stumps without a random_state.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Classes of the target.
    n_classes_ : int
        Number of classes.
    root : RuleTreeNode
        Root of the fitted tree.
    candidate_splits_ : list of tuple
        ``(probability, stump, data_type)``: the stumps usable on the training data, with the
        data type they are used on.
    cache_ : dict
        Values cached by the stumps during fit (see :meth:`get_cache`).

    See Also
    --------
    RuleTreeRegressor : the same tree for regression.
    RuleTree.stumps.classification : the stumps for classification.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from RuleTree import RuleTreeClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> model = RuleTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    >>> model.predict(X[:3])
    array([0, 0, 0])
    """
    def __init__(self,
                 max_leaf_nodes=float('inf'),
                 min_samples_split=2,
                 max_depth=float('inf'),
                 prune_useless_leaves=True,
                 base_stumps: Union[ClassifierMixin, list] = None,
                 stump_selection: str = 'random',
                 random_state=None
                 ):
        """Initialize the RuleTreeClassifier; see the class documentation for the parameters."""
        if base_stumps is None:
            base_stumps = DecisionTreeStumpClassifier()

        super().__init__(max_leaf_nodes=max_leaf_nodes,
                         min_samples_split=min_samples_split,
                         max_depth=max_depth,
                         prune_useless_leaves=prune_useless_leaves,
                         base_stumps=base_stumps,
                         stump_selection=stump_selection,
                         random_state=random_state)

    def check_additional_halting_condition(self, y, curr_idx: np.ndarray):
        """
        Stop splitting a node whose instances all have the same class.

        Parameters
        ----------
        y : numpy.ndarray
            Target of the whole data set.
        curr_idx : numpy.ndarray
            Indices of the instances in the node.

        Returns
        -------
        halt : bool
            True if the node is pure.
        """
        return len(np.unique(y[curr_idx])) == 1  # only 1 target

    def prepare_node(self, y: np.ndarray, idx: np.ndarray, node_id: str, node: Optional[RuleTreeNode] = None) -> RuleTreeNode:
        """
        Create a node (or update ``node``) with the majority class, the class probabilities, the
        log-odds and the number of the instances ``idx``.

        Parameters
        ----------
        y : numpy.ndarray
            Target of the whole data set.
        idx : numpy.ndarray
            Indices of the instances reaching the node.
        node_id : str
            Identifier of the node.
        node : RuleTreeNode, default=None
            Existing node to update instead of creating a new one.

        Returns
        -------
        node : RuleTreeNode
        """
        prediction = calculate_mode(y[idx])
        class_counts = np.array([np.sum(y[idx] == classe) for classe in self.classes_])
        predict_proba = class_counts / len(y[idx])
        node_log_odds = log_odds(class_counts, len(y[idx]))

        if node is not None:
            node.prediction = prediction
            node.prediction_probability = predict_proba
            node.log_odds = node_log_odds
            node.classes = self.classes_
            node.samples = len(y[idx])

            return node


        return RuleTreeNode(
            node_id=node_id,
            prediction=prediction,
            prediction_probability=predict_proba,
            log_odds=node_log_odds,
            classes=self.classes_,
            parent=None,
            stump=None,
            node_l=None,
            node_r=None,
            samples=len(y[idx]),
        )

    def predict_proba(self, X: np.ndarray = None, X_ts=None, X_img=None, X_txt=None):
        """
        Class probabilities of each instance: the ones of the leaf it reaches.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features), default=None
            Tabular data.
        X_ts : numpy.ndarray of shape (n_samples, n_signals, length), default=None
            Time series.
        X_img : numpy.ndarray of shape (n_samples, height, width) or (n_samples, height, width, channels), default=None
            Images, with the channels last (e.g. RGB).
        X_txt : array-like of shape (n_samples,), default=None
            Texts.

        Returns
        -------
        proba : numpy.ndarray of shape (n_samples, n_classes)
            Probabilities of the classes, in the order of ``classes_``.
        """
        _, _, proba = self.root.predict(self._validate_inputs(X=X, X_ts=X_ts, X_img=X_img, X_txt=X_txt),
                                        context=self, cache=getattr(self, "cache_", None))

        return proba


    def _get_stumps_task(self):
        """
        Task of the stumps of the tree.

        Returns
        -------
        task : str
            MODEL_TYPE_CLF.
        """
        return MODEL_TYPE_CLF

    def _node_value(self, node: RuleTreeNode):
        """The value decomposed by local_interpretation: the class probabilities of the node."""
        total = np.sum(node.prediction_probability)
        return node.prediction_probability / (total if not np.isclose(total, 0.0, rtol=1e-09, atol=1e-09) else 1.0)

    def get_balanced_stumps(self, X=None, p=0.2, X_ts=None, X_img=None, X_txt=None):
        """
        Internal nodes whose split is balanced on the given data (e.g. the training set): the smaller side
        has a fraction of all the instances (the balance score) above ``p``. The visit stops at the first
        unbalanced node of each path.

        Parameters
        ----------
        X, X_ts, X_img, X_txt : numpy.ndarray, default=None
            Data, as given to fit.
        p : float, default=0.2
            Minimum balance score.

        Returns
        -------
        stumps : dict
            Node id -> (node, balance score), in depth-first order.
        """
        data = self._validate_inputs(X=X, X_ts=X_ts, X_img=X_img, X_txt=X_txt)
        stumps = {}

        def visit(node):
            if node.is_leaf():
                return
            labels = node.stump.apply(data, context=self, node_id=node.node_id, cache=getattr(self, "cache_", None))
            score = min(np.mean(labels == 1), np.mean(labels == 2))
            if score > p:
                stumps[node.node_id] = (node, score)
                visit(node.node_l)
                visit(node.node_r)

        visit(self.root)
        return stumps
    
    def stumps_to_trees(self, balanced_nodes):
        """
        A tree of depth 1 for each of the given nodes: its split and two leaves.

        Parameters
        ----------
        balanced_nodes : dict
            Node id -> (node, balance score), as returned by :meth:`get_balanced_stumps`.

        Returns
        -------
        trees : dict
            (feature, threshold) of the split -> RuleTreeClassifier with that split as root.
        """
        trees = {}
        for node, _ in balanced_nodes.values():
            tree = self.__class__(**self.get_params(deep=False))
            tree.classes_, tree.n_classes_ = self.classes_, self.n_classes_

            tree.root = copy.deepcopy(node, memo={id(node.parent): None})  # the subtree, without the ancestors
            tree.root.node_l.make_leaf()
            tree.root.node_r.make_leaf()
            tree.root.node_id, tree.root.node_l.node_id, tree.root.node_r.node_id = 'R', 'Rl', 'Rr'
            tree.root.set_stump(tree.root.stump)  # node_id_ of the stump: the root of the new tree

            trees[(node.stump.feature_original, node.stump.threshold_original)] = tree

        return trees
