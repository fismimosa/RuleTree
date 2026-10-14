import heapq
from typing import Union, Optional

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.base import ClusterMixin, ClassifierMixin, RegressorMixin
from sklearn.metrics import r2_score

from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump
from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.regression.DecisionTreeStumpRegressor import DecisionTreeStumpRegressor
from RuleTree.tree.RuleTree import RuleTree
from RuleTree.tree.RuleTreeClassifier import RuleTreeClassifier
from RuleTree.tree.RuleTreeNode import RuleTreeNode
from RuleTree.tree.RuleTreeRegressor import RuleTreeRegressor
from RuleTree.utils import bic, light_famd


class RuleTreeCluster(RuleTree, ClusterMixin):
    """
    Rule Tree Clustering algorithm that uses a decision tree structure for data clustering.
    
    This class implements a clustering algorithm that partitions data using decision rules organized 
    in a tree structure. It works by performing dimensionality reduction (PCA, MCA, or FAMD depending
    on data types) and building a tree where each node represents a rule-based split.
    
    Parameters
    ----------
    n_components : int, default=2
        Number of principal components to use for dimensionality reduction.
    
    clus_impurity : str, default='r2'
        Criterion to evaluate split quality: 
        - 'r2': Use R-squared score
        - 'bic': Use Bayesian Information Criterion
    
    bic_eps : float, default=0.0
        Threshold for determining if a split is useful when using BIC criterion.
    
    max_leaf_nodes : int, default=inf
        Maximum number of leaf nodes in the tree.
        
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
        
    max_depth : int, default=inf
        Maximum depth of the tree.
    
    prune_useless_leaves : bool, default=False
        Whether to prune leaves that don't improve performance.
    
    base_stumps : RegressorMixin or list, default=None
        Regressor to use for building the decision stumps. If None, uses DecisionTreeStumpRegressor.
    
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.
    
    criterion, splitter, min_samples_leaf, min_weight_fraction_leaf, max_features,
    min_impurity_decrease, ccp_alpha, monotonic_cst : 
        Parameters passed to the base decision tree stump regressor if base_stumps is None.
    
    Attributes
    ----------
    label_encoder : dict
        Mapping from original labels to numeric labels.
    """
    def __init__(self,
                 n_components: int = 2,
                 clus_impurity: str = 'r2',
                 bic_eps: float = .0,
                 max_leaf_nodes=float('inf'),
                 min_samples_split=2,
                 max_depth=float('inf'),
                 prune_useless_leaves=False,
                 base_stumps: Union[RegressorMixin, list] = None,
                 random_state=None,
                 distance_measure=None):
        if base_stumps is None:
            base_stumps = DecisionTreeStumpRegressor(max_depth=1)

        super().__init__(max_leaf_nodes=max_leaf_nodes,
                         min_samples_split=min_samples_split,
                         max_depth=max_depth,
                         prune_useless_leaves=prune_useless_leaves,
                         base_stumps=base_stumps,
                         stump_selection='random',
                         random_state=random_state,
                         distance_measure=distance_measure)

        self.n_components = n_components
        self.clus_impurity = clus_impurity
        self.bic_eps = bic_eps

        if self.clus_impurity not in ['bic', 'r2']:
            raise ValueError('Unknown clustering impurity measure %s' % self.clus_impurity)

    def is_split_useless(self, X: np.ndarray|None, X_ts:np.ndarray|None, X_img:np.ndarray|None, X_txt:np.ndarray|None,
                         clf: RuleTreeBaseStump, idx: np.ndarray):
        """
        Determine if a split is useful based on BIC criteria.
        
        Parameters
        ----------
        X : array-like
            Input data samples.
        clf : tree
            Fitted tree classifier/regressor that defines the split.
        idx : ndarray
            Indices of samples in the current node.
            
        Returns
        -------
        bool
            True if the split is considered useless, False otherwise.
        """
        labels = clf.apply(X[idx])

        if len(np.unique(labels)) == 1:
            return True

        # CHECK BIC DECREASE
        bic_parent = bic(X[idx], [0] * len(idx))
        bic_children = bic(X[idx], (np.array(labels) - 1).tolist())

        return bic_parent < bic_children - self.bic_eps * np.abs(bic_parent)

    def fit_predict(self, X, y=None, **kwargs):
        """
        Perform clustering on `X` and returns cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : Ignored
            Not used, present for API consistency by convention.

        **kwargs : dict
            Arguments to be passed to ``fit``.

            .. versionadded:: 1.4

        Returns
        -------
        labels : ndarray of shape (n_samples,), dtype=np.int64
            Cluster labels.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        self.fit(X, **kwargs)
        return self.predict(X)

    def queue_push(self, node: RuleTreeNode, idx: np.ndarray):
        """
        Push a new node into the priority queue for processing.
        
        Parameters
        ----------
        node : RuleTreeNode
            Node to be pushed into the queue.
        idx : ndarray
            Indices of samples in the node.
        """
        heapq.heappush(self.queue, (-len(idx), next(self.tiebreaker), idx, node))

    def make_split(self, X: np.ndarray, y, idx: np.ndarray, **kwargs) -> tree:
        """
        Create a split for the current node using dimensionality reduction.
        
        This method first performs dimensionality reduction on the data using PCA, MCA, 
        or FAMD (depending on data types), then finds the best stump (simple decision tree)
        that splits the data along one of the principal components.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : array-like
            Target values (ignored in clustering, but kept for API consistency).
        idx : ndarray
            Indices of samples in the current node.
        **kwargs : dict
            Additional parameters passed to the function.
            
        Returns
        -------
        tree
            The best decision tree stump for this split.
            
        Raises
        ------
        TypeError
            If the input data has an unsupported shape.
        """
        if len(X.shape) != 2:
            raise TypeError(f'Unsupported data type for shape {X.shape}')

        n_components_split = min(self.n_components, len(idx))

        dtypes = pd.DataFrame(X).infer_objects().dtypes
        numerical = [i for i, dtype in enumerate(dtypes) if pd.api.types.is_numeric_dtype(dtype)]
        categorical = [i for i, dtype in enumerate(dtypes) if not pd.api.types.is_numeric_dtype(dtype)]

        if len(categorical) == 0:  # all continuous
            principal_transform = light_famd.PCA(n_components=n_components_split, random_state=self.random_state)
        elif len(numerical) == 0:  # all categorical
            principal_transform = light_famd.MCA(n_components=n_components_split, random_state=self.random_state)
        else:  # mixed
            principal_transform = light_famd.FAMD(n_components=n_components_split, random_state=self.random_state)

        y_pca = principal_transform.fit_transform(X[idx])
        y_pca_all = np.zeros((X.shape[0], y_pca.shape[1]))
        y_pca_all[idx] = y_pca

        best_clf = None
        best_score = float('inf')
        for i in range(n_components_split):
            clf = self._get_random_stump(y_pca)

            try:
                clf.fit(
                    X=X,
                    y=y_pca_all[:, i],
                    idx=idx,
                    context=self
                )
            except NoSplitFoundWarning:
                continue

            labels = clf.apply(X[idx]).astype(int)
            if self.clus_impurity == 'r2':
                # fraction of the variance of the i-th component explained by the split
                component = y_pca[:, i]
                component_pred = np.zeros_like(component)
                for label in np.unique(labels):
                    component_pred[labels == label] = component[labels == label].mean()
                score = -1 * r2_score(component, component_pred)
            else:
                score = bic(X[idx], (labels - 1).tolist())

            if score < best_score:
                best_score = score
                best_clf = clf

        if best_clf is None:
            raise NoSplitFoundWarning(f"No split found on any of the {n_components_split} components")

        return best_clf


    def compute_medoids(self, X: np.ndarray, y, idx: np.ndarray, **kwargs):
        """
        Compute the medoids for clusters.

        This is a placeholder method for computing cluster medoids.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : array-like
            Target values.
        idx : ndarray
            Indices of samples in the current node.
        **kwargs : dict
            Additional parameters.
        """
        pass
        
    def prepare_node(self, y: np.ndarray, idx: np.ndarray, node_id: str, node: Optional[RuleTreeNode] = None) -> RuleTreeNode:
        """
        Prepare a new node in the tree.

        The node id is used as cluster label during fit, and it is replaced by an integer label in
        :meth:`_post_fit_fix`.

        Parameters
        ----------
        y : ndarray
            Target values.
        idx : ndarray
            Indices of samples in the node.
        node_id : str
            Unique identifier for the node.
        node : RuleTreeNode, default=None
            Existing node to update instead of creating a new one.

        Returns
        -------
        RuleTreeNode
            A new node initialized with the given parameters.
        """
        if node is not None:
            node.prediction = node.node_id
            node.samples = len(idx)
            return node

        return RuleTreeNode(
            node_id=node_id,
            prediction=node_id,
            prediction_probability=-1,
            loss=np.nan,
            classes=np.array(['NA']),
            parent=None,
            stump=None,
            node_l=None,
            node_r=None,
            samples=len(idx),
            log_odds=None
        )

    def _post_fit_fix(self):
        """
        Perform post-fitting operations.

        Replaces the node ids used as cluster labels during fit with integer labels, assigned in
        breadth-first order ('R' -> 0, 'Rl' -> 1, 'Rr' -> 2, ...). Labels are recomputed from the
        node ids, so calling this method more than once is safe.
        """
        node_ids = list(self.root.get_predicates()) + list(self.root.get_leaf_nodes())
        self.label_encoder = {node_id: i for i, node_id in enumerate(sorted(node_ids, key=lambda x: (len(x), x)))}
        self._labels_to_int(self.root)

    def _labels_to_int(self, node: RuleTreeNode):
        """
        Convert node labels to integers recursively.

        Parameters
        ----------
        node : RuleTreeNode
            The current node to process.
        """
        node.prediction = self.label_encoder[node.node_id]

        if node.is_leaf():
            return

        self._labels_to_int(node.node_l)
        self._labels_to_int(node.node_r)

    def _get_stumps_base_class(self):
        """
        Get the base class for the stumps used in the tree.
        
        Returns
        -------
        class
            The RegressorMixin class.
        """
        return RegressorMixin

    def _get_prediction_probas(self, current_node = None, probas=None):
        """
        Get prediction probabilities.
        
        Not implemented for the base clustering class.
        
        Raises
        ------
        NotImplementedError
            Method not implemented in the base class.
        """
        raise NotImplementedError()


class RuleTreeClusterClassifier(RuleTreeCluster, ClassifierMixin):
    """
    Rule Tree Cluster Classifier that combines clustering with classification.
    
    This class extends RuleTreeCluster to perform classification tasks by assigning
    class labels to the discovered clusters.
    
    See RuleTreeCluster for the full list of parameters.
    """
    def prepare_node(self, y: np.ndarray, idx: np.ndarray, node_id: str, node: Optional[RuleTreeNode] = None) -> RuleTreeNode:
        """
        Prepare a new classification node in the tree.

        Parameters
        ----------
        y : ndarray
            Target values.
        idx : ndarray
            Indices of samples in the node.
        node_id : str
            Unique identifier for the node.
        node : RuleTreeNode, default=None
            Existing node to update instead of creating a new one.

        Returns
        -------
        RuleTreeNode
            A new node initialized with classification parameters.
        """
        return RuleTreeClassifier.prepare_node(self, y, idx, node_id, node)

    def _post_fit_fix(self):
        """
        Perform post-fitting operations for classification.

        Nodes already predict class labels, so no post-processing is needed.
        """

    def predict_proba(self, X: np.ndarray):
        """
        Predict class probabilities for the input data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities of the leaf reached by each sample.
        """
        return RuleTreeClassifier.predict_proba(self, X)


class RuleTreeClusterRegressor(RuleTreeCluster, RegressorMixin):
    """
    Rule Tree Cluster Regressor that combines clustering with regression.
    
    This class extends RuleTreeCluster to perform regression tasks by assigning
    continuous values to the discovered clusters.
    
    See RuleTreeCluster for the full list of parameters.
    """
    def prepare_node(self, y: np.ndarray, idx: np.ndarray, node_id: str, node: Optional[RuleTreeNode] = None) -> RuleTreeNode:
        """
        Prepare a new regression node in the tree.
        
        Parameters
        ----------
        y : ndarray
            Target values.
        idx : ndarray
            Indices of samples in the node.
        node_id : str
            Unique identifier for the node.
        node : RuleTreeNode, default=None
            Existing node to update instead of creating a new one.

        Returns
        -------
        RuleTreeNode
            A new node initialized with regression parameters.
        """
        return RuleTreeRegressor.prepare_node(self, y, idx, node_id, node)

    def _post_fit_fix(self):
        """
        Perform post-fitting operations for regression.

        Nodes already predict the mean target value, so no post-processing is needed.
        """
