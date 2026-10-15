"""RegressionSplitter: the best split for regression."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from RuleTree.base.RuleTreeBaseSplitter import RuleTreeBaseSplitter
from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.utils.data_utils import _get_info_gain, get_info_gain
from RuleTree.utils.define import MODEL_TYPE_REG


class RegressionSplitter(RuleTreeBaseSplitter):
    """
    Splitter for regression that maximizes the decrease of impurity.

    The numerical features are split by a scikit-learn ``DecisionTreeRegressor`` of depth 1 with the
    same parameters (``x <= threshold`` goes left). Then each value of each categorical feature is
    tested (``x == value`` goes left), and the best categorical split replaces the numerical one if
    its gain is higher.

    Parameters
    ----------
    criterion : {'squared_error', 'absolute_error', 'poisson'} or callable, default='squared_error'
        Impurity measure. A callable ``f(y_true, y_pred) -> float`` (e.g. a scikit-learn metric) is
        supported only for categorical features.
    min_samples_leaf : int or float, default=1
        Minimum number of instances in each child (a fraction of the instances of the node if
        float). Applies to the numerical features.
    min_weight_fraction_leaf : float, default=0.0
        Minimum fraction of the sample weights of the node in each child. Applies to the numerical
        features.
    max_features : int, float, {'sqrt', 'log2'} or None, default=None
        Number of numerical features considered for the split, drawn at random; None for all.
    random_state : int, RandomState instance or None, default=None
        Seed of the choice of the features (``max_features``) and of the ties between features with
        the same gain. With None, a tree gives the stump its own ``random_state``.
    min_impurity_decrease : float, default=0.0
        A numerical split is kept only if it decreases the weighted impurity by at least this value.

    Attributes
    ----------
    feature_ : int
        Column of the feature matrix used by the split.
    threshold_ : float or object
        Threshold of a numerical split, or value of a categorical split.
    is_categorical_ : bool
        Whether the split is categorical.
    gain_ : float
        Decrease of impurity of the split.
    impurity_ : list of float
        Impurity of the node, of the left child and of the right child.

    See Also
    --------
    ClassificationSplitter : the same splitter for classification.
    DecisionTreeStumpRegressor : the stump made of a MatrixTransformer and this splitter.

    Notes
    -----
    The categorical splits do not use ``sample_weight`` and ignore ``min_samples_leaf``.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from RuleTree.stumps.splitters import RegressionSplitter
    >>> X, y = load_diabetes(return_X_y=True)
    >>> splitter = RegressionSplitter(random_state=0).fit(X, y, numerical=list(range(10)), categorical=[])
    >>> splitter.feature_, round(float(splitter.threshold_), 4)
    (8, -0.0038)
    """

    task = MODEL_TYPE_REG

    def __init__(self, criterion='squared_error', min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features=None, random_state=None, min_impurity_decrease=0.0):
        """Initialize the RegressionSplitter; see the class documentation for the parameters."""
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease

    @classmethod
    def _get_impurity_fun(cls, imp):
        """
        Impurity function of a criterion.

        Parameters
        ----------
        imp : str or callable
            Name of the criterion, or the impurity function itself.

        Returns
        -------
        impurity_fun : callable
            Function (y_true, y_pred) -> impurity.

        Raises
        ------
        NotImplementedError
            For 'friedman_mse'.
        """
        if imp == "squared_error":
            return mean_squared_error
        elif imp == "friedman_mse":
            raise NotImplementedError("not implemented")  # TODO: implement in DecisionTreeRegressor?
        elif imp == "absolute_error":
            return mean_absolute_error
        elif imp == "poisson":
            return mean_poisson_deviance
        else:
            return imp

    @classmethod
    def _impurity_fun(cls, impurity_fun, **x):
        """
        Impurity of y_true predicted as y_pred, 0 if there are no instances.

        Parameters
        ----------
        impurity_fun : str or callable
            Criterion.
        **x : dict
            ``y_true`` and ``y_pred``.

        Returns
        -------
        impurity : float
        """
        f = cls._get_impurity_fun(impurity_fun)
        return f(**x) if len(x["y_true"]) > 0 else 0  # TODO: check

    @classmethod
    def _split_impurity(cls, criterion, y, left):
        """
        Impurity of the node and of the two children, each predicted with its mean.

        Parameters
        ----------
        criterion : str or callable
            Criterion.
        y : numpy.ndarray
            Target of the instances in the node.
        left : numpy.ndarray of bool
            Instances that go to the left child.

        Returns
        -------
        impurity : list of float
            Impurity of the node, of the left child and of the right child (0 for an empty child).
        """
        def impurity(y_true):
            y_pred = np.ones((len(y_true),)) * np.mean(y_true) if len(y_true) > 0 else np.ones((0,))
            return cls._impurity_fun(criterion, y_true=y_true, y_pred=y_pred)

        return [impurity(y), impurity(y[left]), impurity(y[~left])]

    def fit(self, X, y, numerical, categorical, sample_weight=None, check_input=True):
        """
        Find the best split: the threshold of the scikit-learn tree on the numerical features, replaced by a
        categorical split if one has a higher decrease of impurity.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix of the instances in the node.
        y : numpy.ndarray of shape (n_samples,)
            Target of the instances in the node.
        numerical : list of int
            Indices of the numerical columns of X.
        categorical : list of int
            Indices of the categorical columns of X.
        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Sample weights of the instances in the node.
        check_input : bool, default=True
            Whether to check the input data.

        Returns
        -------
        self : RegressionSplitter
            The fitted splitter.

        Raises
        ------
        NoSplitFoundWarning
            If no split is found.
        """
        self.feature_, self.threshold_, self.is_categorical_, self.impurity_ = None, None, False, None
        best_info_gain = -float('inf')

        if len(numerical) > 0:
            tree = DecisionTreeRegressor(max_depth=1, criterion=self.criterion, min_samples_leaf=self.min_samples_leaf,
                                         min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                         max_features=self.max_features, random_state=self.random_state,
                                         min_impurity_decrease=self.min_impurity_decrease)
            tree.fit(X[:, numerical], y, sample_weight=sample_weight, check_input=check_input)

            if tree.tree_.node_count == 1:
                raise NoSplitFoundWarning(f"No split found for X {X.shape} and y {np.unique(y)}")

            self.feature_ = numerical[tree.tree_.feature[0]]
            self.threshold_ = tree.tree_.threshold[0]
            best_info_gain = get_info_gain(tree)
            self.impurity_ = self._split_impurity(self.criterion, y, X[:, self.feature_] <= self.threshold_)

        best_info_gain = self._fit_categorical(X, y, categorical, best_info_gain)

        if self.feature_ is None:
            raise NoSplitFoundWarning(f"No split found for X {X.shape} and y {np.unique(y)}")

        self.gain_ = best_info_gain

        return self

    def _fit_categorical(self, X, y, categorical, best_info_gain):
        """
        Test each value of each categorical feature and keep the split if better than best_info_gain.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : numpy.ndarray
            Target.
        categorical : list of int
            Indices of the categorical columns.
        best_info_gain : float
            Gain of the best split found so far.

        Returns
        -------
        best_info_gain : float
            Gain of the best split.
        """
        len_x = len(X)
        for i in categorical:
            for value in np.unique(X[:, i]):
                X_split = X[:, i] == value
                len_left = np.sum(X_split)
                if len_left == 0 or len_left == len_x:  # degenerate split, one side would be empty
                    continue

                impurity = self._split_impurity(self.criterion, y, X_split)
                info_gain = _get_info_gain(*impurity, len_x, len_left, len_x - len_left)

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    self.feature_ = i
                    self.threshold_ = value
                    self.is_categorical_ = True
                    self.impurity_ = impurity

        return best_info_gain

    def update_statistics(self, X, y, sample_weight=None):
        """
        Recompute the impurity of the node and of the two children.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix of the instances in the node.
        y : numpy.ndarray of shape (n_samples,)
            Target of the instances in the node.
        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Sample weights of the instances in the node.
        """
        self.impurity_ = self._split_impurity(self.criterion, y, self.apply(X) == 1)
