"""MatrixTransformer: the data used as feature matrix."""

import numpy as np

from RuleTree.base.RuleTreeBaseTransformer import RuleTreeBaseTransformer, get_feature_types
from RuleTree.utils.define import DATA_TYPE_IMAGE, DATA_TYPE_TABULAR, DATA_TYPE_TS


class MatrixTransformer(RuleTreeBaseTransformer):
    """
    Uses the data as feature matrix: tabular data as they are, time series and images flattened
    into one feature per value. The feature types (numerical or categorical) are computed on the
    whole data set.
    """

    supported_data_types = (DATA_TYPE_TABULAR, DATA_TYPE_TS, DATA_TYPE_IMAGE)

    def fit(self, X, y, idx=None, data_type=None, cache=None, sample_weight=None):
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
        cache : dict, default=None
            Values shared among the splits of the tree.
        sample_weight : numpy.ndarray, default=None
            Sample weights, all the instances.

        Returns
        -------
        self : MatrixTransformer
        """
        self.numerical_, self.categorical_ = get_feature_types(self.transform(X), cache, data_type)
        return self

    def transform(self, X):
        """
        The data: tabular data as they are, time series and images flattened.

        Parameters
        ----------
        X : numpy.ndarray
            Data of a supported type.

        Returns
        -------
        X_transformed : numpy.ndarray of shape (n_samples, n_features)
        """
        X = np.asarray(X)
        return X if X.ndim <= 2 else X.reshape(X.shape[0], -1)
