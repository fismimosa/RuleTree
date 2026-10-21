"""MatrixTransformer: the data used as feature matrix."""

import numpy as np

from RuleTree.base.RuleTreeBaseTransformer import RuleTreeBaseTransformer, get_feature_types
from RuleTree.utils.define import DATA_TYPE_IMAGE, DATA_TYPE_TABULAR, DATA_TYPE_TS


class MatrixTransformer(RuleTreeBaseTransformer):
    """
    Transformer that uses the data as feature matrix: tabular data as they are, time series and
    images flattened into one feature per value.

    The numerical and categorical columns are inferred once, on the whole data set, and cached in the
    tree. It has no parameters and no fitted state: it is the transformer of the decision tree stumps.

    Attributes
    ----------
    numerical_ : list of int
        Numerical columns of the feature matrix.
    categorical_ : list of int
        Categorical columns of the feature matrix.

    See Also
    --------
    DecisionTreeStumpClassifier, DecisionTreeStumpRegressor : the stumps that use it.

    Examples
    --------
    >>> import numpy as np
    >>> from RuleTree.stumps.transformers import MatrixTransformer
    >>> X_ts = np.zeros((10, 2, 5))  # 10 time series with 2 signals of length 5
    >>> MatrixTransformer().fit(X_ts, None, context=None, node_id=None, cache=None).transform(X_ts).shape
    (10, 10)
    """

    supported_data_types = (DATA_TYPE_TABULAR, DATA_TYPE_TS, DATA_TYPE_IMAGE)

    def fit(self, X, y, idx=None, data_type=None, sample_weight=None, *, context, node_id, cache):
        """
        Compute the feature types of the (flattened) data on the whole data set.

        Parameters
        ----------
        X : numpy.ndarray
            Data of the transformer's type, all the instances.
        y : numpy.ndarray
            Target, all the instances.
        idx : numpy.ndarray, default=None
            Indices of the instances in the node. If None, all the instances.
        data_type : str, default=None
            Type of X, used as key of the cached values.
        sample_weight : numpy.ndarray, default=None
            Sample weights, all the instances.
        context : RuleTree or None
            Tree being built, None outside a tree.
        node_id : str or None
            Identifier of the node, e.g. ``"Rl"`` (``context.get_node_by_id(node_id)``), None outside a tree.
        cache : dict or None
            Cache of the tree (namespace -> dictionary of values), None outside a tree: the component
            decides whether and how to use it (the ones of the library use ``self._get_cache(cache)``).

        Returns
        -------
        self : MatrixTransformer
        """
        X_transformed = self.transform(X, context=context, node_id=node_id, cache=cache)
        self.numerical_, self.categorical_ = get_feature_types(X_transformed, self._get_cache(cache), data_type)
        return self

    def transform(self, X, context=None, node_id=None, cache=None):
        """
        The data: tabular data as they are, time series and images flattened.

        Parameters
        ----------
        X : numpy.ndarray
            Data of a supported type.
        context : RuleTree, default=None
            The tree, None outside a tree.
        node_id : str, default=None
            Identifier of the node of the stump, None outside a tree.
        cache : dict, default=None
            Cache of the tree, when called by it; None otherwise.

        Returns
        -------
        X_transformed : numpy.ndarray of shape (n_samples, n_features)
        """
        X = np.asarray(X)
        return X if X.ndim <= 2 else X.reshape(X.shape[0], -1)
