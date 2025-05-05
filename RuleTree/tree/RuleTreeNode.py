import importlib
from typing import Self

import graphviz
import numpy as np

from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump
from RuleTree.utils.define import GRAPHVIZ_DEFAULT_NODE_SPLIT, GRAPHVIZ_DEFAULT_NODE_LEAF


class RuleTreeNode:
    """
    Represents a node in a Rule Tree.
    
    A Rule Tree node can be either a decision node with a splitting rule (stump) 
    or a leaf node with a prediction. Each node contains information about its prediction,
    its position in the tree hierarchy, and references to its child nodes.
    
    Attributes:
        node_id (str): Unique identifier for the node
        prediction: The predicted class or value
        prediction_probability: Probability distribution across classes or confidence score
        classes: Array of possible class labels
        parent: Reference to parent node
        stump: The splitting rule at this node (None for leaf nodes)
        node_l: Left child node
        node_r: Right child node
        medoids_index: Index of medoid samples (optional)
    """

    def __init__(self,
                 node_id: str,
                 prediction: int | str | float,
                 prediction_probability: np.ndarray | float,
                 classes: np.ndarray,
                 features: np.ndarray,
                 parent: Self | None,
                 stump: RuleTreeBaseStump = None,
                 node_l: Self = None,
                 node_r: Self = None,
                 **kwargs):
        """
        Initialize a RuleTreeNode.
        
        Args:
            node_id: Unique identifier for the node
            prediction: The predicted class or value
            prediction_probability: Probability distribution across classes or confidence score
            classes: Array of possible class labels
            parent: Reference to parent node (None for root)
            stump: The splitting rule at this node (None for leaf nodes)
            node_l: Left child node
            node_r: Right child node
            **kwargs: Additional attributes to be set on the node
        """
        self.node_id = node_id
        self.prediction = prediction
        self.prediction_probability = prediction_probability
        self.classes = classes
        self.features = features
        self.parent = parent
        self.stump = stump
        self.node_l = node_l
        self.node_r = node_r
        self.medoids_index = None

        for name, value in kwargs.items():
            setattr(self, name, value)
            
        
    def is_leaf(self):
        """
        Check if this node is a leaf node.
        
        Returns:
            bool: True if this is a leaf node (has no children), False otherwise
        """
        return self.node_l is None and self.node_r is None

    def make_leaf(self):
        """
        Convert this node into a leaf node by removing its children.
        
        Returns:
            Self: The modified node (now a leaf)
        """
        self.node_l, self.node_r = None, None
        return self

    def simplify(self) -> Self:
        """
        Simplify the tree rooted at this node.
        
        Calls the internal _simplify method and returns the root node.
        
        Returns:
            Self: The root of the simplified tree
        """
        self._simplify()
        return self

    def _simplify(self):  # TODO: update
        """
        Internal method that simplifies the tree by pruning redundant nodes.
        
        Simplifies the tree by removing branches where all leaves have the same prediction.
        
        Returns:
            set: A set of unique prediction values in this subtree
        """
        if self.is_leaf():
            return {self.prediction}
        else:
            all_pred = self.node_l._simplify() | self.node_r._simplify() | {self.prediction}

            if len(all_pred) == 1:
                self.make_leaf()
                return {self.prediction}
            else:
                return all_pred

    def set_stump(self, stump:RuleTreeBaseStump):
        """
        Set the splitting rule (stump) for this node.
        
        Args:
            stump: The RuleTreeBaseStump object representing the splitting rule
        """
        self.stump = stump

    def get_possible_outputs(self) -> tuple[set, set]:  # TODO: update
        """
        Get all possible prediction values in the subtree.
        
        Returns:
            tuple[set, set]: A tuple containing (leaf_predictions, internal_node_predictions)
        """
        if self.is_leaf():
            return {self.prediction}, set()
        else:
            leaf_l, node_l = self.node_l.get_possible_outputs()
            leaf_r, node_r = self.node_r.get_possible_outputs()

            return leaf_l | leaf_r, node_l | node_r | {self.prediction}

    def get_depth(self):
        """
        Get the depth of this node in the tree.
        
        The depth is calculated from the node_id length, which should encode the path.
        
        Returns:
            int: The depth of this node
        """
        return len(self.node_id) - 1

    def get_rule(self, columns_names=None, scaler=None):
        """
        Get a dictionary representation of the rule at this node and its subtree.
        
        Args:
            columns_names: Optional list of feature column names for better rule interpretability
            scaler: Optional scaler for transforming threshold values
            
        Returns:
            dict: Dictionary containing the node information and its rule
        """
        rule = {
            "node_id": self.node_id,
            "is_leaf": self.is_leaf(),
            "prediction": self.prediction,
            "prediction_probability": self.prediction_probability,
            "prediction_classes_": self.classes,
            "features": self.features,
            "left_node": self.node_l.get_rule(columns_names = columns_names, scaler = scaler) if self.node_l is not None else None,
            "right_node": self.node_r.get_rule(columns_names = columns_names, scaler = scaler) if self.node_r is not None else None,
        }
    
        if not self.is_leaf():
            rule |= self.stump.get_rule(columns_names=columns_names, scaler=scaler)
    
        return rule
    
    def node_to_dict(self):
        """
        Convert this node to a dictionary representation for serialization.
        
        The dictionary includes all necessary information to reconstruct the node.
        
        Returns:
            dict: Dictionary representation of the node
        """
        node_as_dict ={
            "node_id": self.node_id,
            "is_leaf": self.is_leaf(),
            "prediction": self.prediction,
            "prediction_probability": self.prediction_probability if isinstance(self.prediction_probability, float) else self.prediction_probability.tolist(),
            "prediction_classes_": self.classes.tolist(),
            "features": self.features,
            "left_node": self.node_l.node_id if self.node_l is not None else None,
            "right_node": self.node_r.node_id if self.node_r is not None else None,
        }

        if not self.is_leaf():
            node_as_dict |= self.stump.node_to_dict()
        
        return node_as_dict

    @classmethod
    def dict_to_node(cls, info_dict, X = None):
        """
        Create a node from its dictionary representation.
        
        Args:
            info_dict: Dictionary containing node information
            X: Optional feature matrix that might be needed for stump reconstruction
            
        Returns:
            RuleTreeNode: Reconstructed node
            
        Raises:
            AssertionError: If required fields are missing in info_dict
        """
        assert 'node_id' in info_dict
        assert 'is_leaf' in info_dict
        if not info_dict['is_leaf']:
            assert 'stump_type' in info_dict

        node = RuleTreeNode(node_id = info_dict['node_id'],
                            prediction = info_dict.get('prediction', np.nan),
                            prediction_probability = info_dict.get('prediction_probability', [np.nan]),
                            parent = None,
                            classes=info_dict.get('prediction_classes_', np.nan),
                            features=info_dict.get('features', np.nan),)
        
        if info_dict['is_leaf'] == True:
            return node

        import_path = info_dict['stump_type']
        print(import_path)
        class_c = getattr(importlib.import_module(import_path), info_dict['stump_type'].split('.')[-1])
        
        node.stump = class_c.dict_to_node(info_dict, X)
        
        return node
    
    def encode_node(self, index, parent, vector, stump, node_index=0):
        """
        Encode the tree into a vector representation.
        
        Used for model serialization in a compact format.
        
        Args:
            index: Dictionary mapping node_id to array indices
            parent: Array tracking parent-child relationships
            vector: 2D array where node information will be stored
            stump: The stump object reference
            node_index: Current index in the encoding
        """
        if self.is_leaf():
            vector[0][node_index] = -1
            vector[1][node_index] = self.prediction
        else:
            feat = self.stump.feature_original[0]
            thr = self.stump.threshold_original[0]
         
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
        Generate a graphviz visualization of the tree.
        
        Args:
            graph: Existing graphviz.Digraph object or None to create a new one
            columns_names: Optional list of feature names for better visualization
            scaler: Optional scaler for transforming threshold values
            float_precision: Number of decimal places for floating point numbers
            
        Returns:
            graphviz.Digraph: The graph visualization object
        """
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

    def predict(self, X: np.ndarray, prediction_step=np.inf):
        """
        Predict labels, leaf indices, and probabilities for the given input data.

        This method traverses the tree recursively to make predictions for each sample
        in the input feature matrix. It stops at leaf nodes or when the specified
        `prediction_step` depth is reached.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        prediction_step : int, optional
            Maximum depth to traverse for prediction. Defaults to np.inf.

        Returns
        -------
        tuple
            A tuple containing:
            - np.ndarray: Predicted labels for each sample.
            - np.ndarray: Leaf node IDs corresponding to each sample.
            - np.ndarray: Predicted probabilities for each sample.
        """
        if self.is_leaf() or prediction_step < 1:
            n = len(X)
            return np.array([self.prediction] * n), \
                np.array([self.node_id] * n), \
                np.array(np.array([self.prediction_probability] * n).reshape(n, -1))

        else:
            labels, leaves, proba = (
                np.full(len(X), fill_value=-1,
                        dtype=object if type(self.prediction) is str else type(self.prediction)),
                np.zeros(len(X), dtype=object),
                np.ones((
                    len(X),
                    1 if type(self.prediction_probability) in [float, int] else len(self.prediction_probability)
                ), dtype=float) * -1
            )

            labels_clf = self.stump.apply(X)
            X_l, X_r = X[labels_clf == 1], X[labels_clf == 2]
            if X_l.shape[0] != 0:
                labels[labels_clf == 1], leaves[labels_clf == 1], proba[labels_clf == 1] = (
                    self.node_l.predict(X_l, prediction_step - 1))
            if X_r.shape[0] != 0:
                labels[labels_clf == 2], leaves[labels_clf == 2], proba[labels_clf == 2] = (
                    self.node_r.predict(X_r, prediction_step - 1))

            return labels, leaves, proba

    def get_predicates(self) -> dict:
        """
        Get the predicates (conditions) for the stump at this node in depth-first order.

        Returns:
            dict: dictionary of predicates of the stump and its children in the form node_id: RuleTreeNode
        """
        if self.is_leaf():
            return dict()
        else:
            return {self.node_id: self} | self.node_l.get_predicates() | self.node_r.get_predicates()

    def get_node_by_id(self, node_id: str) -> Self | None:
        """
        Get a node by its ID in the tree.

        Args:
            node_id: The ID of the node to search for.

        Returns:
            RuleTreeNode: The node with the specified ID or None if not found.
        """
        if self.node_id == node_id:
            return self
        elif self.is_leaf():
            return None
        elif self.node_l is not None:
            return self.node_l.get_node_by_id(node_id)
        elif self.node_r is not None:
            return self.node_r.get_node_by_id(node_id)

    def get_leaf_nodes(self) -> dict[str: Self]:
        """
        Get all leaf nodes in the tree.

        Returns:
            dict: A dictionary of all leaf nodes in the tree.
        """
        if self.is_leaf():
            return {self.node_id: self}
        else:
            return self.node_l.get_leaf_nodes() | self.node_r.get_leaf_nodes()