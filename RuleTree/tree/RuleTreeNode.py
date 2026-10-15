"""RuleTreeNode: a node of a RuleTree."""

import importlib
from typing import Union, Optional

import numpy as np
from sklearn.base import TransformerMixin

from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump
from RuleTree.utils.data_utils import log_odds
from RuleTree.utils.define import GRAPHVIZ_DEFAULT_NODE_SPLIT, GRAPHVIZ_DEFAULT_NODE_LEAF


class RuleTreeNode:
    """
    A node of a RuleTree: a leaf with a prediction, or an internal node with a stump that splits its
    instances between its two children.

    The identifier of a node is its path from the root: ``"R"`` is the root, ``"Rl"`` and ``"Rr"``
    its left and right children, ``"Rlr"`` the right child of ``"Rl"``, and so on.

    Parameters
    ----------
    node_id : str
        Identifier of the node.
    prediction : int, str or float
        Predicted class, value or cluster label.
    prediction_probability : numpy.ndarray or float
        Class probabilities (classification) or standard deviation of the target (regression).
    log_odds : numpy.ndarray or float
        Log-odds of the classes (classification).
    classes : numpy.ndarray or None
        Classes of the target (classification), None otherwise.
    parent : RuleTreeNode or None
        Parent node (None for the root).
    stump : RuleTreeBaseStump, default=None
        Split of the node (None for a leaf).
    node_l, node_r : RuleTreeNode, default=None
        Left and right children (None for a leaf).
    **kwargs : dict
        Other attributes, e.g. ``samples`` (number of instances).

    Attributes
    ----------
    medoids_index : list or None
        Medoids of the node, if computed.
    balance_score : float
        For internal nodes, fraction of the training set in the smaller side of the split.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from RuleTree import RuleTreeClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> model = RuleTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    >>> node = model.get_node_by_id("Rr")
    >>> node.get_depth(), node.samples, node.is_leaf()
    (1, 100, False)
    >>> node.stump.get_rule()["textual_rule"], node.node_l.node_id
    ('X_3 <= 1.75', 'Rrl')
    """

    def __init__(self,
                 node_id: str,
                 prediction: Union[int, str, float],
                 prediction_probability: Union[np.ndarray, float],
                 log_odds: Union[np.ndarray, float],
                 classes: np.ndarray,
                 parent: Optional['RuleTreeNode'],
                 stump: RuleTreeBaseStump = None,
                 node_l: 'RuleTreeNode' = None,
                 node_r: 'RuleTreeNode' = None,
                 **kwargs):
        """Initialize the node; see the class documentation for the parameters."""
        self.node_id = node_id
        self.prediction = prediction
        self.prediction_probability = prediction_probability
        self.log_odds = log_odds
        self.classes = classes
        self.parent = parent
        self.stump = stump
        self.node_l = node_l
        self.node_r = node_r
        self.medoids_index = None

        for name, value in kwargs.items():
            setattr(self, name, value)
            
        
    def is_leaf(self):
        """
        Whether the node is a leaf (it has no children).

        Returns
        -------
        is_leaf : bool
        """
        return self.node_l is None and self.node_r is None

    def make_leaf(self):
        """
        Make the node a leaf by removing its children.

        Returns
        -------
        self : RuleTreeNode
        """
        self.node_l, self.node_r = None, None
        return self

    def simplify(self) -> 'RuleTreeNode':
        """
        Merge the subtrees whose leaves all have the same prediction (see prune_useless_leaves).

        Returns
        -------
        self : RuleTreeNode
            The root of the simplified subtree.
        """
        self._simplify()
        return self

    def _simplify(self):
        """
        Make a leaf of every node whose leaves all have the same prediction.

        Returns
        -------
        predictions : set
            Predictions of the leaves of the subtree.
        """
        if self.is_leaf():
            return {self.prediction}
        else:
            all_pred = self.node_l._simplify() | self.node_r._simplify()

            if len(all_pred) == 1:
                self.make_leaf()
                self.prediction = all_pred.pop()
                return {self.prediction}
            else:
                return all_pred

    def set_stump(self, stump:RuleTreeBaseStump):
        """
        Set the split of the node.

        Parameters
        ----------
        stump : RuleTreeBaseStump
            The fitted stump.
        """
        self.stump = stump

    def get_possible_outputs(self) -> tuple[set, set]:
        """
        Predictions of the nodes of the subtree.

        Returns
        -------
        leaf_predictions : set
            Predictions of the leaves.
        internal_predictions : set
            Predictions of the internal nodes.
        """
        if self.is_leaf():
            return {self.prediction}, set()
        else:
            leaf_l, node_l = self.node_l.get_possible_outputs()
            leaf_r, node_r = self.node_r.get_possible_outputs()

            return leaf_l | leaf_r, node_l | node_r | {self.prediction}

    def get_depth(self):
        """
        Depth of the node, from its identifier (the root has depth 0).

        Returns
        -------
        depth : int
        """
        return len(self.node_id) - 1

    def get_rule(self, columns_names:list=None, scaler:TransformerMixin=None):
        """
        Rules of the subtree, as a nested dictionary (see RuleTree.get_rules).

        Parameters
        ----------
        columns_names : list of str, default=None
            Names of the tabular features.
        scaler : TransformerMixin, default=None
            Scaler used to preprocess the data: the thresholds are shown in the original scale.

        Returns
        -------
        rule : dict
            Statistics of the node (``node_id``, ``is_leaf``, ``prediction``, ``prediction_probability``,
            ``log_odds``, ``prediction_classes_``), the rules of the children (``left_node``,
            ``right_node``) and, for an internal node, the rule of its stump.
        """
        rule = {
            "node_id": self.node_id,
            "is_leaf": self.is_leaf(),
            "prediction": self.prediction,
            "prediction_probability": self.prediction_probability,
            "log_odds": self.log_odds,
            "prediction_classes_": self.classes,
            "left_node": self.node_l.get_rule(columns_names = columns_names, scaler = scaler) if self.node_l is not None else None,
            "right_node": self.node_r.get_rule(columns_names = columns_names, scaler = scaler) if self.node_r is not None else None,
        }
    
        if not self.is_leaf():
            rule |= self.stump.get_rule(columns_names=columns_names, scaler=scaler)
    
        return rule

    def update_statistics(self, data, y, idx=None, context=None):
        """
        Recompute the statistics of classification of the node (majority class, class probabilities,
        log-odds, number of instances) on the instances ``idx``, and the statistics of its stump.
        A node without instances predicts nan.

        Parameters
        ----------
        data : dict
            Whole data set, as ``{data_type: array}``.
        y : numpy.ndarray
            Target of the whole data set.
        idx : numpy.ndarray, default=None
            Indices of the instances reaching the node. If None, all the instances.
        context : RuleTree, default=None
            The tree, whose ``classes_`` are used when the node has none.
        """
        if idx is None:
            idx = slice(None)

        y_local = y[idx]

        classes = None
        if self.classes is not None:
            classes = np.asarray(self.classes)
        elif context is not None and hasattr(context, "classes_") and context.classes_ is not None:
            classes = np.asarray(context.classes_)
        else:
            classes = np.unique(y)

        if classes.ndim == 0:
            classes = classes.reshape(1)

        class_counts = np.array([np.sum(y_local == class_label) for class_label in classes], dtype=float)

        if y_local.shape[0] > 0 and class_counts.sum() > 0:
            self.prediction = classes[np.argmax(class_counts)]
            self.prediction_probability = class_counts / y_local.shape[0]
            self.log_odds = log_odds(class_counts, y_local.shape[0])
        else:
            self.prediction = np.nan
            self.prediction_probability = np.zeros(len(classes), dtype=float)
            self.log_odds = np.full(len(classes), np.nan, dtype=float)

        self.classes = classes
        self.samples = len(y_local)

        if self.stump is not None and not self.is_leaf():
            self.stump.update_statistics(data, y, idx=idx, context=context)
    
    def node_to_dict(self):
        """
        Dictionary representation of the node (without its children, referenced by id).

        Returns
        -------
        node_dict : dict
            JSON-friendly dictionary with the statistics of the node, the ids of its children and, for
            an internal node, the dictionary of its stump.
        """
        node_as_dict ={
            "node_id": self.node_id,
            "is_leaf": self.is_leaf(),
            "prediction": self.prediction,
            "prediction_probability": self.prediction_probability if isinstance(self.prediction_probability, float) else np.array(self.prediction_probability).tolist(),
            "log_odds": self.log_odds if isinstance(self.log_odds, float) else np.array(self.log_odds).tolist(),
            "prediction_classes_": np.array(self.classes).tolist(),
            "left_node": self.node_l.node_id if self.node_l is not None else None,
            "right_node": self.node_r.node_id if self.node_r is not None else None,
        }

        if not self.is_leaf():
            node_as_dict |= self.stump.node_to_dict()
        
        return node_as_dict

    @classmethod
    def dict_to_node(cls, info_dict):
        """
        Create a node from a dictionary created by :meth:`node_to_dict` (children not linked).

        Parameters
        ----------
        info_dict : dict
            Dictionary representation of the node.

        Returns
        -------
        node : RuleTreeNode
        """
        node = RuleTreeNode(node_id=info_dict['node_id'],
                            prediction=info_dict['prediction'],
                            prediction_probability=info_dict['prediction_probability'],
                            log_odds=info_dict['log_odds'],
                            parent=None,
                            classes=info_dict['prediction_classes_'])
        if not info_dict['is_leaf']:
            node.stump = stump_class(info_dict).dict_to_node(info_dict)

        return node

    def encode_node(self, index, parent, vector, stump, node_index=0):
        """
        Encode the subtree in arrays indexed as a complete binary tree (children of i at 2i+1 and 2i+2),
        for trees of univariate splits.

        Parameters
        ----------
        index : dict
            Node id -> position, filled.
        parent : numpy.ndarray
            Position of the parent of each position, filled.
        vector : numpy.ndarray of shape (2, n_positions)
            Feature + 1 (-1 for leaves) and threshold (prediction for leaves) of each position, filled.
        stump : object
            Not used.
        node_index : int, default=0
            Position of this node.
        """
        if self.is_leaf():
            vector[0][node_index] = -1
            vector[1][node_index] = self.prediction
        else:
            feat = self.stump.feature_original
            thr = self.stump.threshold_original
         
            vector[0][node_index] = feat + 1
            vector[1][node_index] = thr

            node_l = self.node_l
            node_r = self.node_r

            index[node_l.node_id] = 2 * node_index + 1
            index[node_r.node_id] = 2 * node_index + 2
            
            parent[2 * index[self.node_id] + 1] = index[self.node_id]
            parent[2 * index[self.node_id] + 2] = index[self.node_id]

            node_l.encode_node(index, parent, vector, stump, 2 * node_index + 1)
            node_r.encode_node(index, parent, vector, stump, 2 * node_index + 2)

    def export_graphviz(self, graph=None, columns_names=None, scaler=None, float_precision=3):
        """
        Graphviz representation of the subtree (requires the graphviz package).

        Parameters
        ----------
        graph : graphviz.Digraph, default=None
            Graph to add the nodes to. If None, a new graph.
        columns_names : list of str, default=None
            Names of the tabular features.
        scaler : TransformerMixin, default=None
            Scaler used to preprocess the data: the thresholds are shown in the original scale.
        float_precision : int, default=3
            Number of decimals of the thresholds.

        Returns
        -------
        graph : graphviz.Digraph
        """
        try:
            import graphviz
        except ImportError:
            raise ImportError('Please install graphviz to visualize tree.')

        if graph is None:
            graph = graphviz.Digraph(name="RuleTree")

        if self.is_leaf():
            graph.node(self.node_id, label=f"{self.prediction}", **GRAPHVIZ_DEFAULT_NODE_LEAF)
            return graph

        rule = self.stump.get_rule(columns_names=columns_names, scaler=scaler, float_precision=float_precision)

        graph.node(self.node_id, **(GRAPHVIZ_DEFAULT_NODE_SPLIT | rule["graphviz_rule"]))

        if self.node_l is not None:
            graph = self.node_l.export_graphviz(graph, columns_names, scaler, float_precision)
            graph.edge(self.node_id, self.node_l.node_id, color="#2ca02c")
        if self.node_r is not None:
            graph = self.node_r.export_graphviz(graph, columns_names, scaler, float_precision)
            graph.edge(self.node_id, self.node_r.node_id, color="#d62728")

        return graph

    def predict(self, data, idx:np.ndarray=None, prediction_step=np.inf):
        """
        Predictions, leaves and probabilities of the instances, traversing the subtree.

        Parameters
        ----------
        data : dict or numpy.ndarray
            Data set, as ``{data_type: array}``. A single array is passed as is to the stumps.
        idx : numpy.ndarray, default=None
            Indices of the instances to predict. If None, all the instances.
        prediction_step : int, default=numpy.inf
            Maximum number of splits to traverse: with a finite value, the prediction of the node
            reached after that many splits.

        Returns
        -------
        labels : numpy.ndarray of shape (n_samples,)
            Prediction of each instance.
        leaves : numpy.ndarray of shape (n_samples,)
            Id of the node reached by each instance.
        proba : numpy.ndarray of shape (n_samples, n_values)
            ``prediction_probability`` of the node reached by each instance.
        """
        if idx is None:
            idx = np.arange(len(next(iter(data.values()))) if isinstance(data, dict) else len(data))
        n = len(idx)

        if self.is_leaf() or prediction_step < 1:
            return np.array([self.prediction] * n), \
                np.array([self.node_id] * n), \
                np.array(np.array([self.prediction_probability] * n).reshape(n, -1))

        else:
            labels, leaves, proba = (
                # isinstance also matches np.str_: a fixed-size string dtype would truncate longer labels
                np.full(n, fill_value=-1, dtype=object if isinstance(self.prediction, str) else type(self.prediction)),
                np.zeros(n, dtype=object),
                np.ones((
                    n,
                    1 if type(self.prediction_probability) in [float, int] else len(self.prediction_probability)
                ), dtype=float) * -1
            )

            labels_clf = self.stump.apply(data, idx)
            if sum(labels_clf == 1) != 0:
                labels[labels_clf == 1], leaves[labels_clf == 1], proba[labels_clf == 1] = (
                    self.node_l.predict(data, idx=idx[labels_clf == 1], prediction_step=prediction_step - 1))
            if sum(labels_clf == 2) != 0:
                labels[labels_clf == 2], leaves[labels_clf == 2], proba[labels_clf == 2] = (
                    self.node_r.predict(data, idx=idx[labels_clf == 2], prediction_step=prediction_step - 1))

            return labels, leaves, proba

    def get_predicates(self) -> dict:
        """
        Internal nodes of the subtree in depth-first order.

        Returns
        -------
        predicates : dict
            Node id -> RuleTreeNode.
        """
        if self.is_leaf():
            return {}
        else:
            return {self.node_id: self} | self.node_l.get_predicates() | self.node_r.get_predicates()

    def get_node_by_id(self, node_id: str) -> Optional['RuleTreeNode']:
        """
        Node of the subtree with the given identifier.

        Parameters
        ----------
        node_id : str
            Identifier of the node.

        Returns
        -------
        node : RuleTreeNode or None
            The node, or None if not in the subtree.
        """
        if self.node_id == node_id:
            return self

        for child in (self.node_l, self.node_r):
            if child is not None:
                node = child.get_node_by_id(node_id)
                if node is not None:
                    return node

        return None

    def get_leaf_nodes(self) -> dict:
        """
        Leaves of the subtree.

        Returns
        -------
        leaves : dict
            Node id -> RuleTreeNode.
        """
        if self.is_leaf():
            return {self.node_id: self}
        else:
            return self.node_l.get_leaf_nodes() | self.node_r.get_leaf_nodes()


def stump_class(stump_dict: dict) -> type:
    """
    Class of a stump saved by node_to_dict: ``stump_class`` in module ``stump_type``.

    Parameters
    ----------
    stump_dict : dict
        Dictionary representation of the stump.

    Returns
    -------
    stump_class : type
    """
    return getattr(importlib.import_module(stump_dict['stump_type']), stump_dict['stump_class'])
