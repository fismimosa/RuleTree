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

    Attributes:
        numerical (list): List of indices for numerical features.
        categorical (list): List of indices for categorical features.
    """

    @abstractmethod
    def fit(self, X, y, idx=None, context=None, sample_weight=None, check_input=True):
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
        dtypes = pd.DataFrame(X).infer_objects().dtypes
        self.numerical = dtypes[dtypes != np.dtype('O')].index
        self.categorical = dtypes[dtypes == np.dtype('O')].index

    @abstractmethod
    def node_to_dict(self) -> dict:
        """
        Abstract method to convert a tree node into a dictionary representation.

        Returns:
            dict: A dictionary representing the node's state.
        """
        raise NotImplementedError("The node_to_dict method must be implemented in subclasses.")

    @abstractmethod
    def apply(self, X: np.ndarray, check_input=True) -> np.ndarray:
        """
        Abstract method to apply the model to input data and return predictions.

        Args:
            X (np.ndarray): Input feature matrix.
            check_input (bool): Whether to check the input data for validity.

        Returns:
            np.ndarray: Predictions made by the model.
        """
        raise NotImplementedError("The apply method must be implemented in subclasses.")

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

    @staticmethod
    def supports(data_type:str) -> bool:
        """
        Checks if the given data type is supported by this model.

        Args:
            data_type (str): Type of input data to check support for.

        Returns:
            bool: True if supported, False otherwise.
        """
        return data_type in [DATA_TYPE_TABULAR]

    @staticmethod
    def update_statistics(self, X, y, idx=None, context=None, sample_weight=None, check_input=True):
        pass
