"""
File: RuleTreeBaseStump.py

This file contains the abstract base class `RuleTreeBaseStump` which defines the structure
and some basic functionalities for a rule-based tree stump model in machine learning.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from RuleTree.utils.define import DATA_TYPE_TABULAR


class RuleTreeBaseStump(BaseEstimator, ABC):
    """
    Abstract base class for a rule-based tree stump model. This class provides the
    structure and some basic functionalities that any rule-based tree stump should implement.

    Data handling
    -------------
    Stumps receive the whole data set as a dictionary ``data = {data_type: array}`` (see
    :data:`RuleTree.utils.define.DATA_INPUT_TYPES`) together with ``idx``, the indices of the
    instances reaching the node. A fitted stump works on a single data type, chosen by the tree
    among ``supported_data_types`` and stored in ``data_type_``. For convenience, ``data`` can
    also be a single array, interpreted as data of the stump's data type.

    The tree passed as ``context`` holds a cache that lives until the next ``fit`` (or
    ``clear_cache()``), where stumps can store anything that can be shared among splits, e.g.
    embeddings computed on the whole data set. Each kind of stump has its own cache, named after
    its class: see :meth:`_get_cache`. Everything needed by ``apply`` must be stored in the stump
    itself.

    Subclasses that work on a 2D matrix can implement ``_transform`` (data of the stump's data
    type -> matrix) and ``_apply_matrix`` (matrix -> 1/2 labels) and reuse :meth:`apply` and
    :meth:`update_statistics`.

    Attributes:
        supported_data_types (tuple): Data types the stump can be fitted on.
        data_type_ (str): Data type the stump has been fitted on.
        numerical (list): List of indices for numerical features.
        categorical (list): List of indices for categorical features.
    """

    supported_data_types = (DATA_TYPE_TABULAR,)

    @abstractmethod
    def fit(self, data, y, idx=None, context=None, data_type=None, sample_weight=None, check_input=True):
        """
        Fit the stump on the instances ``idx`` of ``data``.

        Args:
            data (dict | np.ndarray): Whole data set, as ``{data_type: array}``.
            y (np.ndarray): Target of the whole data set.
            idx (np.ndarray, optional): Indices of the instances reaching the node. Defaults to all.
            context (RuleTree, optional): Tree being built, used to share cached values among splits.
            data_type (str, optional): Data type to use. Defaults to the first supported type in data.
            sample_weight (np.ndarray, optional): Sample weights of the whole data set.
            check_input (bool): Whether to check the input data.

        Returns:
            RuleTreeBaseStump: The fitted stump.
        """
        raise NotImplementedError("The fit method must be implemented in subclasses.")

    @abstractmethod
    def get_rule(self, columns_names:list=None, scaler:TransformerMixin=None, float_precision: Optional[int] = 3) -> dict:
        """
        Abstract method to generate a rule based on the model's learned parameters.

        Args:
            columns_names (list, optional): List of column names for which to generate rules. Defaults to None.
            scaler (object, optional): Scaler object used for preprocessing data. Defaults to None.
            float_precision (int|None, optional): Precision for floating-point numbers in the rule output. Defaults to 3.

        Returns:
            dict: A dictionary representing the generated rule.
        """
        pass

    @abstractmethod
    def node_to_dict(self) -> dict:
        """
        Abstract method to convert a tree node into a dictionary representation.

        Returns:
            dict: A dictionary representing the node's state.
        """
        raise NotImplementedError("The node_to_dict method must be implemented in subclasses.")

    @classmethod
    @abstractmethod
    def dict_to_node(cls, node_dict:dict, X:np.ndarray) -> None:
        """
        Abstract method to create a tree node from its dictionary representation.

        Args:
            node_dict (dict): Dictionary containing the state of the node.
            X (pd.DataFrame): Input feature matrix used for reconstruction.

        Returns:
            RuleTreeBaseStump: An instance of RuleTreeBaseStump or its subclass.
        """
        raise NotImplementedError("The dict_to_node method must be implemented in subclasses.")

    @classmethod
    def supports(cls, data_type:str) -> bool:
        """
        Checks if the given data type is supported by this model.

        Args:
            data_type (str): Type of input data to check support for.

        Returns:
            bool: True if supported, False otherwise.
        """
        return data_type in cls.supported_data_types

    def apply(self, data, idx=None) -> np.ndarray:
        """
        Apply the split to the instances ``idx`` of ``data``.

        Args:
            data (dict | np.ndarray): Data set, as ``{data_type: array}``.
            idx (np.ndarray, optional): Indices of the instances to split. Defaults to all.

        Returns:
            np.ndarray: For each instance in idx, 1 if it goes to the left child, 2 otherwise.
        """
        X = self._get_data(data)
        if idx is not None:
            X = X[idx]

        return self._apply_matrix(self._transform(X))

    def update_statistics(self, data, y, idx=None, context=None, sample_weight=None, check_input=True):
        """
        Recompute the statistics of the split (e.g. impurity) on the instances ``idx`` of ``data``,
        without changing the split.
        """
        if idx is None:
            idx = slice(None)
        X = self._transform(self._get_data(data)[idx])

        self._update_statistics_matrix(X, y[idx], None if sample_weight is None else sample_weight[idx])

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Turn data of the stump's data type into the matrix the split is applied on."""
        return X

    def _apply_matrix(self, X: np.ndarray) -> np.ndarray:
        """Apply the split to the matrix returned by :meth:`_transform`."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement apply or _apply_matrix.")

    def _update_statistics_matrix(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        """Recompute the split statistics on the matrix returned by :meth:`_transform`."""

    def _resolve_fit_data(self, data, data_type=None) -> tuple[np.ndarray, str]:
        """
        Select the data to fit on and store its type in ``data_type_``.

        Args:
            data (dict | np.ndarray): Data set, as ``{data_type: array}`` or a single array.
            data_type (str, optional): Data type to use. Defaults to the first supported type in data.

        Returns:
            tuple[np.ndarray, str]: Data of the selected type (all the instances) and its type.

        Raises:
            ValueError: If the data type is not supported or not available in data.
        """
        if not isinstance(data, dict):
            data = {data_type if data_type is not None else self.supported_data_types[0]: data}

        if data_type is None:
            available = [t for t in self.supported_data_types if t in data]
            if not available:
                raise ValueError(f"{self.__class__.__name__} supports {list(self.supported_data_types)} data, "
                                 f"got {list(data)}.")
            data_type = available[0]

        if not self.supports(data_type):
            raise ValueError(f"{self.__class__.__name__} does not support {data_type} data.")
        if data_type not in data:
            raise ValueError(f"No {data_type} data provided.")

        self.data_type_ = data_type
        return np.asarray(data[data_type]), data_type

    def _get_data(self, data) -> np.ndarray:
        """Return the data of the type the stump has been fitted on."""
        return np.asarray(data[self.data_type_]) if isinstance(data, dict) else np.asarray(data)

    def _set_feature_types(self, X: np.ndarray, context=None) -> None:
        """
        Set ``numerical`` and ``categorical`` for X, the data of the stump's data type (all the
        instances). The result is cached in the tree, so it is computed once per data type.
        """
        cache = self._get_cache(context)
        key = ("feature_types", self.data_type_)
        if key not in cache:
            self.feature_analysis(X, None)
            cache[key] = (self.numerical, self.categorical)

        self.numerical, self.categorical = cache[key]

    def _get_cache(self, context) -> dict:
        """
        Cache shared by the stumps of this class in the tree ``context`` (namespace: the class name).
        Without a context, an empty dictionary that is not kept.
        """
        return context.get_cache(self.__class__.__name__) if context is not None else {}

    def _restore_data_type(self, node_dict: dict) -> None:
        """Set ``data_type_`` from a dictionary created by node_to_dict (older ones have none)."""
        self.data_type_ = node_dict.get("data_type", self.supported_data_types[0])

    def _restore_split(self, node_dict: dict) -> None:
        """
        Set the split (``feature_original``, ``threshold_original``, ``is_categorical``, ``data_type_``)
        from a dictionary created by node_to_dict. Unfitted stumps have no split.
        """
        self.feature_original = None
        if 'feature_idx' in node_dict:
            self.feature_original = np.array([node_dict['feature_idx'], -2, -2], dtype=int)

        self.threshold_original = None
        if 'threshold' in node_dict:
            threshold = node_dict['threshold']
            # categorical splits compare with a value of the feature, e.g. a string
            self.threshold_original = np.array([threshold, -2, -2], dtype=float if _is_number(threshold) else object)

        self.is_categorical = node_dict.get('is_categorical', False)
        self._restore_data_type(node_dict)

    def feature_analysis(self, X:np.ndarray, y:np.ndarray) -> None:
        """
        Analyzes features of input data and categorizes them into numerical and categorical types.

        Args:
            X (pd.DataFrame): Input feature matrix.
            y (pd.Series): Target labels.

        Sets the following attributes:
            self.numerical (list): Indices of numerical features.
            self.categorical (list): Indices of categorical features.
        """
        if np.issubdtype(np.asarray(X).dtype, np.number):  # fast path, e.g. for time series and images
            self.numerical = list(range(X.shape[1]))
            self.categorical = []
            return

        dtypes = pd.DataFrame(X).infer_objects().dtypes
        self.numerical = [i for i, x in enumerate(dtypes) if pd.api.types.is_numeric_dtype(x)]
        self.categorical = [i for i, x in enumerate(dtypes) if not pd.api.types.is_numeric_dtype(x)]


def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
