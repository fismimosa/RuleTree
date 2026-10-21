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
    prune_useless_leaves : bool, default=False
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
                 prune_useless_leaves=False,
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
        X_img : numpy.ndarray of shape (n_samples, ...), default=None
            Images.
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
        """
        Value of a node in _get_prediction_probas: its class probabilities.

        Parameters
        ----------
        node : RuleTreeNode
            The node.

        Returns
        -------
        probabilities : numpy.ndarray
        """
        return node.prediction_probability

    def local_interpretation(self, X, joint_contribution = False):
        """
        Contribution of each feature to the predicted class probabilities, as in treeinterpreter: the
        probabilities of an instance are the ones of the root (bias) plus the contributions of the
        features of the splits along its path.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Tabular data.
        joint_contribution : bool, default=False
            If True, the contributions are given to the sets of features along the path.

        Returns
        -------
        prediction : numpy.ndarray of shape (n_samples, n_classes)
            Class probabilities of the leaf of each instance.
        bias : numpy.ndarray of shape (n_samples, n_classes)
            Class probabilities of the root.
        contributions : numpy.ndarray of shape (n_samples, n_features, n_classes) or list of dict
            Contribution of each feature (or set of features, with joint_contribution).
        """
        leaves, paths, leaf_to_path, values = super().local_interpretation(X = X,
                                                                           joint_contribution = joint_contribution)
        normalizer = values.sum(axis=1)[:, np.newaxis]
        normalizer[np.isclose(normalizer, 0.0, rtol=1e-09, atol=1e-09)] = 1.0
        values /= normalizer

        biases = np.tile(values[paths[0][0]], (X.shape[0], 1))
        line_shape = (X.shape[1], self.n_classes_)
        
        return super().eval_contributions(
                                        leaves=leaves,
                                        paths=paths,
                                        leaf_to_path=leaf_to_path,
                                        values=values,
                                        biases=biases,
                                        line_shape=line_shape,
                                        joint_contribution=joint_contribution
                                    )
   

    
    def get_balanced_stumps(self, current_node=None, stumps=None, p=0.2):
        """
        Internal nodes whose split is balanced: the smaller side has a fraction of the training set
        (``balance_score``) above ``p``. The visit stops at the first unbalanced node of each path.

        Parameters
        ----------
        current_node : RuleTreeNode, default=None
            Root of the subtree. If None, the root of the tree.
        stumps : dict, default=None
            Dictionary being filled.
        p : float, default=0.2
            Minimum balance score.

        Returns
        -------
        stumps : dict
            Node id -> (node, balance score).
        """
        if stumps is None:
            stumps = {}
        if current_node is None:
            current_node = self.root
            
        if not current_node.is_leaf() and current_node.balance_score > p:
                stumps[current_node.node_id] = (current_node, current_node.balance_score)
                self.get_balanced_stumps(current_node=current_node.node_l, stumps=stumps, p=p)
                self.get_balanced_stumps(current_node=current_node.node_r, stumps=stumps, p=p)
                
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

    @classmethod
    def complete_tree(cls, node, X, y, n_classes_):
        """
        Recompute recursively the predictions, class probabilities and number of instances of the
        nodes of a tree on new data, with classes ``0, ..., n_classes_ - 1``.

        Parameters
        ----------
        node : RuleTreeNode
            Root of the subtree.
        X : numpy.ndarray
            Data reaching the node.
        y : numpy.ndarray
            Target of the data reaching the node.
        n_classes_ : int
            Number of classes.
        """
        classes_ = list(range(n_classes_))
        node.prediction = calculate_mode(y)
        node.prediction_probability = np.zeros((len(classes_), ))
        node.samples = len(y)
        
        
        for i, classe in enumerate(classes_):
            node.prediction_probability[i] = np.sum(np.where(y == classe, 1, 0)) / len(y)

        if not node.is_leaf():            
            labels_clf = node.stump.apply(X, node_id=node.node_id)
            X_l, X_r = X[labels_clf == 1], X[labels_clf == 2]
            y_l, y_r = y[labels_clf == 1], y[labels_clf == 2]         
            if X_l.shape[0] != 0:
                cls.complete_tree(node.node_l, X_l, y_l, n_classes_)
            if X_r.shape[0] != 0:
                cls.complete_tree(node.node_r, X_r, y_r, n_classes_)
