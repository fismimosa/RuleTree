"""
File: RuleTreeBaseTransformer.py

Base class of the transformers: the component of a stump that turns the data of one type
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
    Turns the data of one type into a feature matrix.

    A transformer is fitted on the whole data set of its type (``X``, all the instances) and on
    ``idx``, the instances reaching the node, so that values computed on the whole data set can be
    shared among the splits through ``cache`` (a dictionary that lives as long as the tree's cache).

    After the splitter has chosen a feature, ``select`` lets the transformer keep only what is needed
    to compute that feature (e.g. a single pivot); if the feature matrix changes, ``refit_after_select``
    is True and the stump fits the splitter again on the new matrix.

    Attributes set by fit:
        numerical_ (list): Indices of the numerical columns of the feature matrix.
        categorical_ (list): Indices of the categorical columns of the feature matrix.
    """

    supported_data_types = (DATA_TYPE_TABULAR,)

    # whether select changes the feature matrix, so that the splitter must be fitted again
    refit_after_select = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, idx: np.ndarray = None, data_type: str = None, cache: dict = None,
            sample_weight=None):
        """
        Fit the transformer.

        Args:
            X (np.ndarray): Data of the transformer's type, all the instances.
            y (np.ndarray): Target, all the instances.
            idx (np.ndarray, optional): Indices of the instances in the node. Defaults to all.
            data_type (str, optional): Type of X, used as key of the cached values.
            cache (dict, optional): Values shared among the splits of the tree.
            sample_weight (np.ndarray, optional): Sample weights, all the instances.

        Returns:
            RuleTreeBaseTransformer: The fitted transformer.
        """

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Feature matrix of the instances in X (data of the transformer's type)."""

    def select(self, feature: int) -> int:
        """
        Keep only what is needed to compute ``feature``, the column chosen by the splitter.

        Returns:
            int: Index of the feature in the matrix returned by transform from now on.
        """
        return feature

    def feature_id(self, feature: int):
        """Identifier of a feature of the matrix in the rules (e.g. a column of the data, a pivot)."""
        return feature

    def get_state(self) -> dict:
        """Fitted state needed by transform, as a JSON-friendly dictionary (see set_state)."""
        return {}

    def set_state(self, state: dict):
        """Restore the fitted state saved by get_state."""
        return self


def get_feature_types(X: np.ndarray, cache: dict = None, data_type: str = None) -> tuple[list, list]:
    """
    Indices of the numerical and of the categorical columns of a 2D matrix. When a cache is given,
    the result is stored under ("feature_types", data_type), so it is computed once per data type.
    """
    cache = {} if cache is None else cache
    key = ("feature_types", data_type)
    if key not in cache:
        cache[key] = _feature_types(X)

    return cache[key]


def _feature_types(X: np.ndarray) -> tuple[list, list]:
    if np.issubdtype(np.asarray(X).dtype, np.number):  # fast path, e.g. for time series and images
        return list(range(X.shape[1])), []

    dtypes = pd.DataFrame(X).infer_objects().dtypes
    numerical = [i for i, x in enumerate(dtypes) if pd.api.types.is_numeric_dtype(x)]
    categorical = [i for i, x in enumerate(dtypes) if not pd.api.types.is_numeric_dtype(x)]
    return numerical, categorical
