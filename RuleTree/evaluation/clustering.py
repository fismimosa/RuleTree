"""Metrics of a clustering: agreement with known classes, and quality without them."""

import numpy as np
import sklearn.metrics as skm


def evaluate_clustering(labels_true, labels_pred) -> dict:
    """
    Agreement between a clustering and known classes (external metrics).

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Known classes.
    labels_pred : array-like of shape (n_samples,)
        Cluster labels.

    Returns
    -------
    scores : dict
        ``adjusted_rand``, ``rand_score``, ``adjusted_mutual_info``, ``mutual_info``,
        ``normalized_mutual_info``, ``homogeneity``, ``completeness``, ``v_measure`` and
        ``fowlkes_mallows``.

    See Also
    --------
    evaluate_clustering_internal : metrics that do not need the classes.

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> from RuleTree import RuleTreeCluster
    >>> from RuleTree.evaluation import evaluate_clustering
    >>> X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)
    >>> labels = RuleTreeCluster(max_depth=3, random_state=0).fit_predict(X)
    >>> round(evaluate_clustering(y, labels)["adjusted_rand"], 2)
    0.95
    """
    return {
        "adjusted_rand": skm.adjusted_rand_score(labels_true, labels_pred),
        "rand_score": skm.rand_score(labels_true, labels_pred),
        "adjusted_mutual_info": skm.adjusted_mutual_info_score(labels_true, labels_pred),
        "mutual_info": skm.mutual_info_score(labels_true, labels_pred),
        "normalized_mutual_info": skm.normalized_mutual_info_score(labels_true, labels_pred),
        "homogeneity": skm.homogeneity_score(labels_true, labels_pred),
        "completeness": skm.completeness_score(labels_true, labels_pred),
        "v_measure": skm.v_measure_score(labels_true, labels_pred),
        "fowlkes_mallows": skm.fowlkes_mallows_score(labels_true, labels_pred),
    }


def evaluate_clustering_internal(X, labels, metric="euclidean") -> dict:
    """
    Quality of a clustering from the data alone (internal metrics).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features), or (n_samples, n_samples) if metric='precomputed'
        Numerical data, or the matrix of the distances between the instances.
    labels : array-like of shape (n_samples,)
        Cluster labels.
    metric : str or callable, default='euclidean'
        Distance used by the silhouette, as in ``sklearn.metrics.silhouette_score``; 'precomputed'
        if X is a distance matrix (then only the silhouette is computed).

    Returns
    -------
    scores : dict
        ``silhouette_score`` (the higher the better), ``calinski_harabasz`` (the higher the better)
        and ``davies_bouldin`` (the lower the better); nan with a single cluster.

    See Also
    --------
    evaluate_clustering : metrics that compare the clusters with known classes.

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> from RuleTree import RuleTreeCluster
    >>> from RuleTree.evaluation import evaluate_clustering_internal
    >>> X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)
    >>> labels = RuleTreeCluster(max_depth=3, random_state=0).fit_predict(X)
    >>> round(evaluate_clustering_internal(X, labels)["silhouette_score"], 2)
    0.59
    """
    if len(np.unique(labels)) < 2:
        return {"silhouette_score": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}

    precomputed = metric == "precomputed"
    return {
        "silhouette_score": skm.silhouette_score(X, labels, metric=metric),
        "calinski_harabasz": np.nan if precomputed else skm.calinski_harabasz_score(X, labels),
        "davies_bouldin": np.nan if precomputed else skm.davies_bouldin_score(X, labels),
    }
