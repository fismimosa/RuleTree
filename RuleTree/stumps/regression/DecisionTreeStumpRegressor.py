import copy
import warnings

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._criterion import FriedmanMSE

from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump
from RuleTree.exceptions import NoSplitFoundWarning
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
        rule["impurity"] = self.impurity

        rule["args"] |= self.kwargs

        rule["split"] = {
            "args": {}
        }

        return rule

    @classmethod
    def dict_to_node(cls, node_dict, X=None):
        """
         Deserializes a dictionary into a DecisionTreeStumpRegressor node.

        This is essentially the inverse operation of node_to_dict(), recreating
        a stump instance from a serialized dictionary representation.
        
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
        self = cls()

        if 'feature_idx' in node_dict:
            self.feature_original = np.ones(3, dtype=int) * -2
            self.feature_original[0] = node_dict.get('feature_idx', -2)

        self.threshold_original = None
        if 'threshold' in node_dict:
            self.threshold_original = np.ones(3) * -2
            self.threshold_original[0] = node_dict.get('threshold', -2)

        self.is_categorical = node_dict.get('is_categorical', None)

        args = copy.deepcopy(node_dict.get("args", {}))
        self.kwargs = args

        self.__set_impurity_fun(args["criterion"])

        return self

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
        """
        super().__init__(**kwargs)
        self.is_categorical = False
        self.kwargs = kwargs
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
            raise NotImplementedError("not implemented") # TODO: implement in DecisionTreeRegressor?
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

            #no split
            if len(self.feature_original) == 1:
                raise NoSplitFoundWarning(f"No split found for X {X.shape} and y {np.unique(y)}")

            curr_pred = np.ones((len(y),)) * np.mean(y)
            idx_l = X[:, self.feature_original[0]] <= self.threshold_original[0]
            idx_r = X[:, self.feature_original[0]] > self.threshold_original[0]
            l_pred = np.ones((len(y[idx_l]),)) * np.mean(y[idx_l])
            r_pred = np.ones((len(y[idx_r]),)) * np.mean(y[idx_r])
            self.impurity = [
                self._impurity_fun(self.impurity_fun, y_true=y, y_pred=curr_pred),
                self._impurity_fun(self.impurity_fun, y_true=y[idx_l], y_pred=l_pred),
                self._impurity_fun(self.impurity_fun, y_true=y[idx_r], y_pred=r_pred),
            ]
            
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
                        self.is_categorical = True
                        self.impurity = [
                            self._impurity_fun(self.impurity_fun, y_true=y, y_pred=curr_pred),
                            self._impurity_fun(self.impurity_fun, y_true=y[X_split[:, 0]], y_pred=l_pred),
                            self._impurity_fun(self.impurity_fun, y_true=y[~X_split[:, 0]], y_pred=r_pred),
                        ]


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

    def update_statistics(self, X, y, idx=None, context=None, sample_weight=None, check_input=True):
        X = X[idx]
        y = y[idx]

        curr_pred = np.ones((len(y),)) * np.mean(y)
        if self.is_categorical:
            idx_l = X[:, self.feature_original[0]] == self.threshold_original[0]
            idx_r = X[:, self.feature_original[0]] != self.threshold_original[0]
        else:
            idx_l = X[:, self.feature_original[0]] <= self.threshold_original[0]
            idx_r = X[:, self.feature_original[0]] > self.threshold_original[0]
        l_pred = np.ones((len(y[idx_l]),)) * np.mean(y[idx_l])
        r_pred = np.ones((len(y[idx_r]),)) * np.mean(y[idx_r])
        self.impurity = [
            self._impurity_fun(self.impurity_fun, y_true=y, y_pred=curr_pred),
            self._impurity_fun(self.impurity_fun, y_true=y[idx_l], y_pred=l_pred),
            self._impurity_fun(self.impurity_fun, y_true=y[idx_r], y_pred=r_pred),
        ]