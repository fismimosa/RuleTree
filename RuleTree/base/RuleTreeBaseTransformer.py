"""
Abstract base class of the transformers: the component of a stump that turns the data of one type
(tabular data, time series, images, ...) into the feature matrix split by the stump's splitter
(see RuleTreeBaseSplitter).
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from RuleTree.utils.define import DATA_TYPE_TABULAR


class RuleTreeBaseTransformer(BaseEstimator, ABC):
    """
    Abstract base class of the transformers, the component of a stump that turns the data of one
    type into a feature matrix.

    A new transformer implements :meth:`transform` and, if it learns something from the data,
    :meth:`fit` (and :meth:`get_state`/:meth:`set_state` to be saved with the tree).

    A transformer is fitted on the whole data set of its type (``X``, all the instances) and on
    ``idx``, the instances reaching the node, so that values computed on the whole data set can be
    shared among the splits through ``cache`` (a dictionary that lives as long as the tree's cache).

    After the splitter has chosen a feature, :meth:`select` lets the transformer keep only what is
    needed to compute that feature (e.g. a single pivot); if the feature matrix changes,
    ``refit_after_select`` is True and the stump fits the splitter again on the new matrix.

    Transformers are scikit-learn estimators: their parameters are the arguments of ``__init__``.

    Attributes
    ----------
    supported_data_types : tuple of str
        Data types the transformer can be fitted on. Class attribute.
    refit_after_select : bool
        Whether :meth:`select` changes the feature matrix. Class attribute.
    numerical_ : list of int
        Indices of the numerical columns of the feature matrix. Set by :meth:`fit`.
    categorical_ : list of int
        Indices of the categorical columns of the feature matrix. Set by :meth:`fit`.

    See Also
    --------
    RuleTree.stumps.Stump.Stump : the stump that combines a transformer and a splitter.
    RuleTreeBaseSplitter : the other component of a stump.

    Examples
    --------
    A transformer without fitted state defines only ``transform`` (see :doc:`/guides/extending`):

    >>> import numpy as np
    >>> from RuleTree import RuleTreeClassifier
    >>> from RuleTree.base.RuleTreeBaseTransformer import RuleTreeBaseTransformer
    >>> from RuleTree.stumps.Stump import Stump
    >>> from RuleTree.stumps.splitters import ClassificationSplitter
    >>> from RuleTree.utils.define import DATA_TYPE_TS
    >>> class MeanTransformer(RuleTreeBaseTransformer):
    ...     supported_data_types = (DATA_TYPE_TS,)
    ...
    ...     def transform(self, X):
    ...         return X.mean(axis=2)  # mean of each signal
    ...
    ...     def feature_id(self, feature):
    ...         return f"mean_{feature}"
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 2, size=50)
    >>> X_ts = rng.normal(size=(50, 1, 20)) + 2 * y[:, None, None]
    >>> stump = Stump(MeanTransformer(), ClassificationSplitter())
    >>> model = RuleTreeClassifier(max_depth=1, base_stumps=stump).fit(X_ts=X_ts, y=y)
    >>> model.get_rules()["textual_rule"].split()[0]
    'mean_0'
    """

    supported_data_types = (DATA_TYPE_TABULAR,)

    # whether select changes the feature matrix, so that the splitter must be fitted again
    refit_after_select = False

    def fit(self, X: np.ndarray, y: np.ndarray, idx: np.ndarray = None, data_type: str = None, cache: dict = None,
            sample_weight=None):
        """
        Fit the transformer and set ``numerical_`` and ``categorical_``.

        The default implementation, for transformers without fitted state, only sets the types of the
        columns of ``transform(X)``: such a transformer implements just :meth:`transform`.

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
        self : RuleTreeBaseTransformer
            The fitted transformer.

        Raises
        ------
        NoSplitFoundWarning
            If the data of the node cannot be transformed into features (e.g. no numerical feature).
        """
        self.numerical_, self.categorical_ = get_feature_types(self.transform(X))
        return self

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Feature matrix of the instances in X.

        Parameters
        ----------
        X : numpy.ndarray
            Data of the transformer's type.

        Returns
        -------
        X_transformed : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix.
        """

    def select(self, feature: int) -> int:
        """
        Keep only what is needed to compute ``feature``, the column chosen by the splitter.
        The default implementation keeps everything.

        Parameters
        ----------
        feature : int
            Column of the feature matrix chosen by the splitter.

        Returns
        -------
        feature : int
            Index of the feature in the matrix returned by :meth:`transform` from now on.
        """
        return feature

    def feature_id(self, feature: int):
        """
        Identifier of a feature of the matrix in the rules, e.g. a column of the data or a pivot.
        The default implementation is the index of the column.

        Parameters
        ----------
        feature : int
            Column of the feature matrix.

        Returns
        -------
        feature_id : int or str
            Identifier shown in the rules (e.g. ``X_3``) and stored in ``feature_original``.
        """
        return feature

    def get_state(self) -> dict:
        """
        Fitted state needed by :meth:`transform`. The default implementation is an empty state.

        Returns
        -------
        state : dict
            JSON-friendly dictionary, restored by :meth:`set_state`.
        """
        return {}

    def set_state(self, state: dict):
        """
        Restore the fitted state saved by :meth:`get_state`.

        Parameters
        ----------
        state : dict
            Dictionary created by :meth:`get_state`.

        Returns
        -------
        self : RuleTreeBaseTransformer
        """
        return self


def get_feature_types(X: np.ndarray, cache: dict = None, data_type: str = None) -> tuple[list, list]:
    """
    Indices of the numerical and of the categorical columns of a 2D matrix.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Feature matrix.
    cache : dict, default=None
        If given, the result is stored under ``("feature_types", data_type)``, so that it is
        computed once per data type.
    data_type : str, default=None
        Type of the data X comes from.

    Returns
    -------
    numerical : list of int
        Indices of the numerical columns.
    categorical : list of int
        Indices of the categorical columns.
    """
    cache = {} if cache is None else cache
    key = ("feature_types", data_type)
    if key not in cache:
        cache[key] = _feature_types(X)

    return cache[key]


def _feature_types(X: np.ndarray) -> tuple[list, list]:
    """
    Numerical and categorical columns of X, inferred with pandas (numbers are numerical, anything
    else is categorical).

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Feature matrix.

    Returns
    -------
    numerical : list of int
    categorical : list of int
    """
    if np.issubdtype(np.asarray(X).dtype, np.number):  # fast path, e.g. for time series and images
        return list(range(X.shape[1])), []

    dtypes = pd.DataFrame(X).infer_objects().dtypes
    numerical = [i for i, x in enumerate(dtypes) if pd.api.types.is_numeric_dtype(x)]
    categorical = [i for i, x in enumerate(dtypes) if not pd.api.types.is_numeric_dtype(x)]
    return numerical, categorical
