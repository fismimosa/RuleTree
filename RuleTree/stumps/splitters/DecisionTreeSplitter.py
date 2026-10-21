"""DecisionTreeSplitter: what ClassificationSplitter and RegressionSplitter have in common."""

from abc import ABC, abstractmethod

import numpy as np

from RuleTree.base.RuleTreeBaseSplitter import RuleTreeBaseSplitter
from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.utils.data_utils import _get_info_gain, get_info_gain


class DecisionTreeSplitter(RuleTreeBaseSplitter, ABC):
    """
    Base class of the splitters that maximize the decrease of impurity, ClassificationSplitter and
    RegressionSplitter.

    The numerical features are split by a scikit-learn decision tree of depth 1 (``_sklearn_tree``)
    with the same parameters (``x <= threshold`` goes left); on a binary feature, the threshold is
    then moved halfway between its two values. Then each value of each categorical feature is tested
    (``x == value`` goes left), skipping the values that would leave a child empty, and the best
    categorical split replaces the numerical one if its gain is higher.

    A subclass defines ``task``, ``_sklearn_tree`` and :meth:`_impurities`, and, if the scikit-learn
    tree has more parameters, its ``__init__`` and :meth:`_tree_params`.

    Parameters
    ----------
    criterion : str or callable
        Impurity measure (see the subclasses).
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
        Decrease of impurity (information gain) of the split.
    impurity_ : list of float
        Impurity of the node, of the left child and of the right child.

    Notes
    -----
    The categorical splits do not use ``sample_weight`` and ignore ``min_samples_leaf``.
    """

    # scikit-learn tree (DecisionTreeClassifier or DecisionTreeRegressor) that splits the numerical features
    _sklearn_tree = None

    def __init__(self, criterion, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                 random_state=None, min_impurity_decrease=0.0):
        """Initialize the splitter; see the class documentation for the parameters."""
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease

    def _tree_params(self) -> dict:
        """
        Parameters of the scikit-learn tree besides those of this class.

        Returns
        -------
        params : dict
        """
        return {}

    @abstractmethod
    def _impurities(self, y: np.ndarray, left: np.ndarray, sample_weight=None) -> list:
        """
        Impurity of the node and of the two children of a split.

        Parameters
        ----------
        y : numpy.ndarray
            Target of the instances in the node.
        left : numpy.ndarray of bool
            Instances that go to the left child.
        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Sample weights of the instances in the node.

        Returns
        -------
        impurity : list of float
            Impurity of the node, of the left child and of the right child.
        """

    def fit(self, X, y, numerical, categorical, sample_weight=None, check_input=True, *, context,
            node_id):
        """
        Find the best split: the threshold of the scikit-learn tree on the numerical features (halfway
        between the two values of a binary feature), replaced by a categorical split if one has a higher
        decrease of impurity.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix of the instances in the node.
        y : numpy.ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target of the instances in the node.
        numerical : list of int
            Indices of the numerical columns of X.
        categorical : list of int
            Indices of the categorical columns of X.
        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Sample weights of the instances in the node.
        check_input : bool, default=True
            Whether to check the input data.
        context : RuleTree, default=None
            Tree being built, None outside a tree.
        node_id : str, default=None
            Identifier of the node, e.g. ``"Rl"`` (``context.get_node_by_id(node_id)``), None outside a tree.

        Returns
        -------
        self : DecisionTreeSplitter
            The fitted splitter.

        Raises
        ------
        NoSplitFoundWarning
            If no split is found.
        """
        self.feature_, self.threshold_, self.is_categorical_, self.impurity_ = None, None, False, None
        self.gain_ = -float('inf')

        if len(numerical) > 0:
            tree = self._sklearn_tree(max_depth=1, criterion=self.criterion, min_samples_leaf=self.min_samples_leaf,
                                      min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                      max_features=self.max_features, random_state=self.random_state,
                                      min_impurity_decrease=self.min_impurity_decrease, **self._tree_params())
            tree.fit(X[:, numerical], y, sample_weight=sample_weight, check_input=check_input)

            if tree.tree_.node_count > 1:  # otherwise only the categorical features can split
                self.feature_ = numerical[tree.tree_.feature[0]]
                self.threshold_ = tree.tree_.threshold[0]
                self.gain_ = get_info_gain(tree)
                self.impurity_ = self._impurities(y, X[:, self.feature_] <= self.threshold_, sample_weight)

        self._fit_categorical(X, y, categorical)

        if self.feature_ is None:
            raise NoSplitFoundWarning(f"No split found for X {X.shape} and y {np.unique(y)}")

        X_feature = X[:, self.feature_]
        if not self.is_categorical_ and len(np.unique(X_feature)) == 2:
            # binary feature: threshold halfway between the two values
            self.threshold_ = (X_feature.max() + X_feature.min()) / 2

        return self

    def _fit_categorical(self, X, y, categorical):
        """
        Test each value of each categorical feature and keep the split if its gain is higher than gain_.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : numpy.ndarray
            Target.
        categorical : list of int
            Indices of the categorical columns.
        """
        len_x = len(X)
        for i in categorical:
            for value in np.unique(X[:, i]):
                X_split = X[:, i] == value
                len_left = np.sum(X_split)
                if len_left == 0 or len_left == len_x:  # degenerate split, one side would be empty
                    continue

                impurity = self._impurities(y, X_split)
                info_gain = _get_info_gain(*impurity, len_x, len_left, len_x - len_left)

                if info_gain > self.gain_:
                    self.gain_ = info_gain
                    self.feature_ = i
                    self.threshold_ = value
                    self.is_categorical_ = True
                    self.impurity_ = impurity

    def update_statistics(self, X, y, sample_weight=None, *, context, node_id):
        """
        Recompute the impurity of the node and of the two children.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix of the instances in the node.
        y : numpy.ndarray
            Target of the instances in the node.
        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Sample weights of the instances in the node.
        context : RuleTree or None
            The tree, None outside a tree.
        node_id : str or None
            Identifier of the node, e.g. ``"Rl"`` (``context.get_node_by_id(node_id)``), None outside a tree.
        """
        self.impurity_ = self._impurities(y, self.apply(X, context=context, node_id=node_id) == 1, sample_weight)
