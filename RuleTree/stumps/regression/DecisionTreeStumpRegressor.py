import copy
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance
from sklearn.tree import DecisionTreeRegressor

from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump
from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier

from RuleTree.utils.data_utils import get_info_gain, _get_info_gain


class DecisionTreeStumpRegressor(DecisionTreeRegressor, RuleTreeBaseStump):
    """
    A decision tree stump regressor that implements a single-level decision tree.
    
    This class extends both scikit-learn's DecisionTreeRegressor and RuleTreeBaseStump
    to provide functionality for creating single-level decision trees for regression tasks
    that can handle both numerical and categorical features. It supports extracting rules 
    and serialization to/from dictionary format.
    
    Parameters
    ----------
    **kwargs : dict
        Additional parameters to pass to scikit-learn's DecisionTreeRegressor.
        Notable parameters include:
        - criterion: Function to measure the quality of a split ("squared_error", 
          "friedman_mse", "absolute_error", "poisson", default="squared_error")
        - max_depth: Maximum depth of the tree (default=1 for stumps)
    """
    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        """
        Extract the decision rule from the stump in human-readable format.
        
        Parameters
        ----------
        columns_names : array-like, optional
            Names of the features. If provided, feature names are used instead of indices.
        scaler : object, optional
            Scaler object used to transform features. If provided, threshold values are
            transformed back to the original scale.
        float_precision : int, optional
            Number of decimal places to round threshold values to.
            
        Returns
        -------
        dict
            Dictionary containing rule information including feature index, threshold,
            textual representation of the rule, and visualization properties.
        """
        return DecisionTreeStumpClassifier.get_rule(self,
                                                    columns_names=columns_names,
                                                    scaler=scaler,
                                                    float_precision=float_precision)

    def node_to_dict(self):
        """
        Convert the stump to a dictionary representation for serialization.
        
        Returns
        -------
        dict
            Dictionary representation of the stump including all parameters
            needed to reconstruct it.
        """
        rule = self.get_rule(float_precision=None)

        rule["stump_type"] = self.__class__.__name__
        rule["samples"] = self.tree_.n_node_samples[0]
        rule["impurity"] = self.tree_.impurity[0]

        rule["args"] = {
                           "unique_val_enum": self.unique_val_enum,
                       } | self.kwargs

        rule["split"] = {
            "args": {}
        }

        return rule

    def dict_to_node(self, node_dict, X=None):
        """
        Create a stump from its dictionary representation.
        
        Parameters
        ----------
        node_dict : dict
            Dictionary containing the stump parameters.
        X : array-like, optional
            Input data that may be used for additional fitting.
            
        Returns
        -------
        None
        """
        assert 'feature_idx' in node_dict
        assert 'threshold' in node_dict
        assert 'is_categorical' in node_dict

        self.feature_original = np.zeros(3)
        self.threshold_original = np.zeros(3)

        self.feature_original[0] = node_dict["feature_idx"]
        self.threshold_original[0] = node_dict["threshold"]
        self.is_categorical = node_dict["is_categorical"]

        args = copy.deepcopy(node_dict.get("args", dict()))
        self.unique_val_enum = args.pop("unique_val_enum", np.nan)
        self.kwargs = args

        self.__set_impurity_fun(args["criterion"])

    def __init__(self, **kwargs):
        """
        Initialize a new DecisionTreeStumpRegressor.
        
        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to scikit-learn's DecisionTreeRegressor.
            Notable parameters include:
            - criterion: Function to measure the quality of a split ("squared_error", 
              "friedman_mse", "absolute_error", "poisson", default="squared_error")
            - max_depth: Maximum depth of the tree (default=1 for stumps)
        """
        super().__init__(**kwargs)
        self.is_categorical = False
        self.kwargs = kwargs
        self.unique_val_enum = None
        self.threshold_original = None
        self.feature_original = None

        self.impurity_fun = kwargs['criterion'] if 'criterion' in kwargs else "squared_error"

    @classmethod
    def _get_impurity_fun(cls, imp):
        """
        Get the appropriate impurity function based on the criterion name.
        
        Parameters
        ----------
        imp : str or callable
            The name of the impurity criterion or a callable function.
            
        Returns
        -------
        callable
            The impurity function to use for evaluating splits.
            
        Raises
        ------
        Exception
            If an unimplemented criterion is requested.
        """
        if imp == "squared_error":
            return mean_squared_error
        elif imp == "friedman_mse":
            raise Exception("not implemented") # TODO: implement
        elif imp == "absolute_error":
            return mean_absolute_error
        elif imp == "poisson":
            return mean_poisson_deviance
        else:
            return imp


    @classmethod
    def _impurity_fun(cls, impurity_fun, **x):
        """
        Apply the impurity function to the provided data.
        
        Parameters
        ----------
        impurity_fun : str or callable
            The name of the impurity criterion or a callable function.
        **x : dict
            Arguments to pass to the impurity function.
            
        Returns
        -------
        float
            The calculated impurity value.
        """
        f = cls._get_impurity_fun(impurity_fun)
        return f(**x) if len(x["y_true"]) > 0 else 0 # TODO: check

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            Not used, kept for API consistency.
            
        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        return self.kwargs

    def fit(self, X, y, idx=None, context=None, sample_weight=None, check_input=True):
        """
        Build a decision stump by fitting to the input data.
        
        The method first tries to find the best split with numerical features using
        scikit-learn's implementation, then checks if categorical features might provide
        a better split.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (continuous values).
        idx : array-like, optional
            Indices of samples to use for training.
        context : object, optional
            Additional context information that may be used during fitting.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
        check_input : bool, default=True
            Whether to check input consistency.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        if idx is None:
            idx = slice(None)
        X = X[idx]
        y = y[idx]

        self.feature_analysis(X, y)
        best_info_gain = -float('inf')

        if len(self.numerical) > 0:
            super().fit(X[:, self.numerical], y, sample_weight=sample_weight, check_input=check_input)
            self.feature_original = [self.numerical[x] if x != -2 else x for x in self.tree_.feature]
            self.threshold_original = self.tree_.threshold
            self.n_node_samples = self.tree_.n_node_samples
            best_info_gain = get_info_gain(self)
            
        self._fit_cat(X, y, best_info_gain)

        return self

    def _fit_cat(self, X, y, best_info_gain):
        """
        Find the best split using categorical features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        best_info_gain : float
            Current best information gain from numerical features.
        """
        if self.max_depth > 1:
            raise Exception("not implemented") # TODO: implement?

        len_x = len(X)

        if len(self.categorical) > 0 and best_info_gain != float('inf'):
            for i in self.categorical:
                for value in np.unique(X[:, i]):
                    X_split = X[:, i:i+1] == value
                    len_left = np.sum(X_split)
                    curr_pred = np.ones((len(y), ))*np.mean(y)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        l_pred = np.ones((len(y[X_split[:, 0]]),)) * np.mean(y[X_split[:, 0]])
                        r_pred = np.ones((len(y[~X_split[:, 0]]),)) * np.mean(y[~X_split[:, 0]])

                        info_gain = _get_info_gain(self._impurity_fun(self.impurity_fun, y_true=y, y_pred=curr_pred),
                                                   self._impurity_fun(self.impurity_fun, y_true=y[X_split[:, 0]], y_pred=l_pred),
                                                   self._impurity_fun(self.impurity_fun, y_true=y[~X_split[:, 0]], y_pred=r_pred),
                                                   len_x,
                                                   len_left,
                                                   len_x - len_left)

                    if info_gain > best_info_gain:
                        best_info_gain = info_gain
                        self.feature_original = [i, -2, -2]
                        self.threshold_original = np.array([value, -2, -2])
                        self.unique_val_enum = np.unique(X[:, i])
                        self.is_categorical = True


    def apply(self, X, check_input=False):
        """
        Apply the decision stump to X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        check_input : bool, default=False
            Whether to check input consistency.
            
        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted node indices (1 for left node, 2 for right node).
        """
        if len(self.feature_original) < 3:
            return np.ones(X.shape[0])

        if not self.is_categorical:
            y_pred = np.ones(X.shape[0], dtype=int) * 2
            X_feature = X[:, self.feature_original[0]]
            y_pred[X_feature <= self.threshold_original[0]] = 1
            
            return y_pred
        else:
            y_pred = np.ones(X.shape[0], dtype=int) * 2
            X_feature = X[:, self.feature_original[0]]
            y_pred[X_feature == self.threshold_original[0]] = 1

            return y_pred
