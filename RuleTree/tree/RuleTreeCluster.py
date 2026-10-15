"""
RuleTreeCluster and its variants: RuleTrees for clustering, also used to classify or regress
with clusters found without looking at the target.
"""

import heapq
from typing import Union, Optional

import numpy as np
import pandas as pd
from sklearn.base import ClusterMixin, ClassifierMixin, RegressorMixin
from sklearn.metrics import r2_score

from RuleTree.utils.define import MODEL_TYPE_REG
from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump
from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.stumps.regression.DecisionTreeStumpRegressor import DecisionTreeStumpRegressor
from RuleTree.tree.RuleTree import RuleTree
from RuleTree.tree.RuleTreeClassifier import RuleTreeClassifier
from RuleTree.tree.RuleTreeNode import RuleTreeNode
from RuleTree.tree.RuleTreeRegressor import RuleTreeRegressor
from RuleTree.utils import bic, light_famd
from RuleTree.utils.define import DATA_TYPE_TABULAR


class RuleTreeCluster(ClusterMixin, RuleTree):
    """
    RuleTree for clustering: each leaf is a cluster described by the rules of its path.

    To split a node, the instances are projected on their principal components (PCA, MCA or FAMD,
    for numerical, categorical or mixed data); for each component, a regression stump is fitted to
    predict it, and the best split is kept according to ``clus_impurity``. A node is split only if
    the split decreases its BIC (see ``bic_eps``). The tree grows from the largest node. Only
    tabular data are supported.

    After fit, the nodes predict integer labels assigned in breadth-first order ('R' -> 0,
    'Rl' -> 1, 'Rr' -> 2, ...); ``predict`` returns the label of the leaf of each instance.

    Parameters
    ----------
    n_components : int, default=2
        Number of principal components predicted by the stumps of each node.
    clus_impurity : {'r2', 'bic'}, default='r2'
        Criterion to choose the split among the ones on the different components: the variance of the
        component explained by the split ('r2') or the Bayesian Information Criterion of the two
        clusters ('bic').
    bic_eps : float, default=0.0
        A split is kept only if it decreases the BIC of the node by more than bic_eps times its value.
    max_leaf_nodes : int, default=inf
        Maximum number of leaves.
    min_samples_split : int, default=2
        Minimum number of instances to split a node.
    max_depth : int, default=inf
        Maximum depth of the tree (the root has depth 0).
    prune_useless_leaves : bool, default=False
        Whether to merge, after fit, the sibling leaves with the same prediction.
    base_stumps : RuleTreeBaseStump or list, default=None
        Regression stumps used to split the nodes, fitted to predict a principal component. If None,
        DecisionTreeStumpRegressor().
    random_state : int, RandomState instance or None, default=None
        Seed of the principal component analysis and of the random choice of the stumps, also given
        to the stumps without a random_state.
    distance_measure : str or callable, default=None
        Not used.

    Attributes
    ----------
    label_encoder : dict
        Node id -> integer label.
    root : RuleTreeNode
        Root of the fitted tree.
    candidate_splits_ : list of tuple
        ``(probability, stump, data_type)``: the stumps usable on the training data, with the
        data type they are used on.
    cache_ : dict
        Values cached by the stumps during fit (see :meth:`get_cache`).

    See Also
    --------
    RuleTreeClusterClassifier, RuleTreeClusterRegressor : the clustering tree with leaves that predict
        a target.

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.metrics import adjusted_rand_score
    >>> from RuleTree import RuleTreeCluster
    >>> X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)
    >>> labels = RuleTreeCluster(max_depth=3, random_state=0).fit_predict(X)
    >>> round(adjusted_rand_score(y, labels), 2)
    0.95
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
        """Initialize the RuleTreeCluster; see the class documentation for the parameters."""
        if base_stumps is None:
            base_stumps = DecisionTreeStumpRegressor()

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

    def is_split_useless(self, data: dict, idx: np.ndarray, labels: np.ndarray):
        """
        Whether a split is useless: all the instances go to the same child, or the BIC of the two
        clusters is not lower than the BIC of the node (by bic_eps times its value).

        Parameters
        ----------
        data : dict
            Whole data set, as ``{data_type: array}``.
        idx : numpy.ndarray
            Indices of the instances in the node.
        labels : numpy.ndarray
            Result of the split on idx (1 left, 2 right).

        Returns
        -------
        useless : bool
        """
        if len(np.unique(labels)) == 1:
            return True

        # CHECK BIC DECREASE
        X = data[DATA_TYPE_TABULAR]
        bic_parent = bic(X[idx], [0] * len(idx))
        bic_children = bic(X[idx], (np.array(labels) - 1).tolist())

        return bic_parent < bic_children - self.bic_eps * np.abs(bic_parent)

    def _set_candidate_splits(self, data: dict):
        """
        Candidate splits, as in RuleTree; clustering supports only tabular data.

        Parameters
        ----------
        data : dict
            Whole data set, as ``{data_type: array}``.

        Raises
        ------
        ValueError
            If data other than tabular are provided.
        """
        if set(data) != {DATA_TYPE_TABULAR}:
            raise ValueError(f"{self.__class__.__name__} supports only tabular data (X), got {', '.join(data)}.")

        super()._set_candidate_splits(data)

    def fit_predict(self, X, y=None, **kwargs):
        """
        Fit the tree and return the cluster label of each instance.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Tabular data.
        y : None
            Not used, present for API consistency.
        **kwargs : dict
            Arguments passed to fit.

        Returns
        -------
        labels : numpy.ndarray of shape (n_samples,)
            Cluster labels.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        self.fit(X, **kwargs)
        return self.predict(X)

    def queue_push(self, node: RuleTreeNode, idx: np.ndarray):
        """
        Push a node into the queue, with priority to the nodes with more instances.

        Parameters
        ----------
        node : RuleTreeNode
            The node.
        idx : numpy.ndarray
            Indices of the instances reaching the node.
        """
        heapq.heappush(self.queue, (-len(idx), next(self.tiebreaker), idx, node))

    def make_split(self, data: dict, y, idx: np.ndarray, sample_weight=None) -> RuleTreeBaseStump:
        """
        Split a node on its principal components: the instances are projected with PCA, MCA or FAMD
        (numerical, categorical or mixed data), a stump is fitted to predict each component, and the
        best split according to ``clus_impurity`` is kept.

        Parameters
        ----------
        data : dict
            Whole data set, as ``{data_type: array}`` (only tabular data).
        y : numpy.ndarray or None
            Not used by the split (the stumps predict the components).
        idx : numpy.ndarray
            Indices of the instances in the node.
        sample_weight : None
            Not used, present for API consistency.

        Returns
        -------
        stump : RuleTreeBaseStump
            The best stump.

        Raises
        ------
        NoSplitFoundWarning
            If no component can be split.
        """
        X = data[DATA_TYPE_TABULAR]
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
            clf, data_type = self._get_random_split()

            try:
                clf.fit(data, y_pca_all[:, i], idx=idx, context=self, data_type=data_type)
            except NoSplitFoundWarning:
                continue

            labels = clf.apply(data, idx)
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
        Not implemented for clustering: does nothing.

        Parameters
        ----------
        X : numpy.ndarray
            Not used.
        y : numpy.ndarray
            Not used.
        idx : numpy.ndarray
            Not used.
        **kwargs : dict
            Not used.
        """
        pass
        
    def prepare_node(self, y: np.ndarray, idx: np.ndarray, node_id: str, node: Optional[RuleTreeNode] = None) -> RuleTreeNode:
        """
        Create a node (or update ``node``) whose prediction is its node id, replaced by an integer label
        in :meth:`_post_fit_fix`.

        Parameters
        ----------
        y : numpy.ndarray or None
            Not used.
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

    def _update_node_statistics(self, node, data, y, idx):
        """
        Only the number of instances: the predictions of the nodes are the cluster labels.

        Parameters
        ----------
        node : RuleTreeNode
            The node.
        data : dict
            Not used.
        y : numpy.ndarray or None
            Not used.
        idx : numpy.ndarray
            Indices of the instances reaching the node.
        """
        node.samples = len(idx)

    def _post_fit_fix(self):
        """
        Replace the node ids used as cluster labels during fit with integer labels, assigned in
        breadth-first order ('R' -> 0, 'Rl' -> 1, 'Rr' -> 2, ...). Labels are recomputed from the node
        ids, so calling this method more than once is safe.
        """
        node_ids = list(self.root.get_predicates()) + list(self.root.get_leaf_nodes())
        self.label_encoder = {node_id: i for i, node_id in enumerate(sorted(node_ids, key=lambda x: (len(x), x)))}
        self._labels_to_int(self.root)

    def _labels_to_int(self, node: RuleTreeNode):
        """
        Replace recursively the prediction of each node with its integer label.

        Parameters
        ----------
        node : RuleTreeNode
            Root of the subtree.
        """
        node.prediction = self.label_encoder[node.node_id]

        if node.is_leaf():
            return

        self._labels_to_int(node.node_l)
        self._labels_to_int(node.node_r)

    def _get_stumps_task(self):
        """
        Task of the stumps of the tree: they predict the principal components.

        Returns
        -------
        task : str
            MODEL_TYPE_REG.
        """
        return MODEL_TYPE_REG

    def _get_prediction_probas(self, current_node = None, probas=None):
        """
        Not implemented for clustering.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError()


class RuleTreeClusterClassifier(ClassifierMixin, RuleTreeCluster):
    """
    RuleTreeCluster whose leaves predict a class: the tree is built as in RuleTreeCluster, without
    looking at the target, and each node predicts the majority class of its instances, as in
    RuleTreeClassifier.

    Parameters
    ----------
    n_components : int, default=2
        Number of principal components predicted by the stumps of each node.
    clus_impurity : {'r2', 'bic'}, default='r2'
        Criterion to choose the split among the ones on the different components: the variance of the
        component explained by the split ('r2') or the Bayesian Information Criterion of the two
        clusters ('bic').
    bic_eps : float, default=0.0
        A split is kept only if it decreases the BIC of the node by more than bic_eps times its value.
    max_leaf_nodes : int, default=inf
        Maximum number of leaves.
    min_samples_split : int, default=2
        Minimum number of instances to split a node.
    max_depth : int, default=inf
        Maximum depth of the tree (the root has depth 0).
    prune_useless_leaves : bool, default=False
        Whether to merge, after fit, the sibling leaves with the same prediction.
    base_stumps : RuleTreeBaseStump or list, default=None
        Regression stumps used to split the nodes, fitted to predict a principal component. If None,
        DecisionTreeStumpRegressor().
    random_state : int, RandomState instance or None, default=None
        Seed of the principal component analysis and of the random choice of the stumps, also given
        to the stumps without a random_state.
    distance_measure : str or callable, default=None
        Not used.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Classes of the target.
    n_classes_ : int
        Number of classes.

    See Also
    --------
    RuleTreeCluster : the clustering tree.
    RuleTreeClassifier : a tree whose splits are chosen with the target.

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> from RuleTree.tree.RuleTreeCluster import RuleTreeClusterClassifier
    >>> X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)
    >>> model = RuleTreeClusterClassifier(max_depth=2, random_state=0).fit(X, y)
    >>> round(model.score(X, y), 2)
    0.99
    """
    def prepare_node(self, y: np.ndarray, idx: np.ndarray, node_id: str, node: Optional[RuleTreeNode] = None) -> RuleTreeNode:
        """
        Create a node (or update ``node``) with the statistics of classification, as in RuleTreeClassifier.

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
        return RuleTreeClassifier.prepare_node(self, y, idx, node_id, node)

    def _post_fit_fix(self):
        """Nothing to do: the nodes already predict class labels."""

    def _update_node_statistics(self, node, data, y, idx):
        """
        Statistics of classification, as in RuleTreeClassifier.

        Parameters
        ----------
        node : RuleTreeNode
            The node.
        data : dict
            Whole data set, as ``{data_type: array}``.
        y : numpy.ndarray
            Target of the whole data set.
        idx : numpy.ndarray
            Indices of the instances reaching the node.
        """
        RuleTree._update_node_statistics(self, node, data, y, idx)

    def predict_proba(self, X: np.ndarray = None, X_ts=None, X_img=None, X_txt=None):
        """
        Class probabilities of each instance: the ones of the leaf it reaches.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features), default=None
            Tabular data.
        X_ts, X_img, X_txt : default=None
            Not supported by clustering, present for API consistency.

        Returns
        -------
        proba : numpy.ndarray of shape (n_samples, n_classes)
        """
        return RuleTreeClassifier.predict_proba(self, X=X, X_ts=X_ts, X_img=X_img, X_txt=X_txt)


class RuleTreeClusterRegressor(RegressorMixin, RuleTreeCluster):
    """
    RuleTreeCluster whose leaves predict a value: the tree is built as in RuleTreeCluster, without
    looking at the target, and each node predicts the mean of the target of its instances, as in
    RuleTreeRegressor.

    Parameters
    ----------
    n_components : int, default=2
        Number of principal components predicted by the stumps of each node.
    clus_impurity : {'r2', 'bic'}, default='r2'
        Criterion to choose the split among the ones on the different components: the variance of the
        component explained by the split ('r2') or the Bayesian Information Criterion of the two
        clusters ('bic').
    bic_eps : float, default=0.0
        A split is kept only if it decreases the BIC of the node by more than bic_eps times its value.
    max_leaf_nodes : int, default=inf
        Maximum number of leaves.
    min_samples_split : int, default=2
        Minimum number of instances to split a node.
    max_depth : int, default=inf
        Maximum depth of the tree (the root has depth 0).
    prune_useless_leaves : bool, default=False
        Whether to merge, after fit, the sibling leaves with the same prediction.
    base_stumps : RuleTreeBaseStump or list, default=None
        Regression stumps used to split the nodes, fitted to predict a principal component. If None,
        DecisionTreeStumpRegressor().
    random_state : int, RandomState instance or None, default=None
        Seed of the principal component analysis and of the random choice of the stumps, also given
        to the stumps without a random_state.
    distance_measure : str or callable, default=None
        Not used.

    Attributes
    ----------
    root : RuleTreeNode
        Root of the fitted tree.

    See Also
    --------
    RuleTreeCluster : the clustering tree.
    RuleTreeRegressor : a tree whose splits are chosen with the target.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from RuleTree.tree.RuleTreeCluster import RuleTreeClusterRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> model = RuleTreeClusterRegressor(max_depth=2, random_state=0).fit(X, y)
    >>> model.predict(X[:2]).round(1)
    array([177.6, 111.8])
    """
    def prepare_node(self, y: np.ndarray, idx: np.ndarray, node_id: str, node: Optional[RuleTreeNode] = None) -> RuleTreeNode:
        """
        Create a node (or update ``node``) with the statistics of regression, as in RuleTreeRegressor.

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
        return RuleTreeRegressor.prepare_node(self, y, idx, node_id, node)

    def _post_fit_fix(self):
        """Nothing to do: the nodes already predict the mean of the target."""

    def _update_node_statistics(self, node, data, y, idx):
        """
        Statistics of regression, as in RuleTreeRegressor.

        Parameters
        ----------
        node : RuleTreeNode
            The node.
        data : dict
            Whole data set, as ``{data_type: array}``.
        y : numpy.ndarray
            Target of the whole data set.
        idx : numpy.ndarray
            Indices of the instances reaching the node.
        """
        RuleTreeRegressor._update_node_statistics(self, node, data, y, idx)
