"""ClassificationSplitter: the best split for classification."""

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from RuleTree.base.RuleTreeBaseSplitter import RuleTreeBaseSplitter
from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.utils.data_utils import _get_info_gain, entropy, get_info_gain, gini
from RuleTree.utils.define import MODEL_TYPE_CLF


class ClassificationSplitter(RuleTreeBaseSplitter):
    """
    Splitter for classification that maximizes the decrease of impurity (information gain).

    The numerical features are split by a scikit-learn ``DecisionTreeClassifier`` of depth 1 with the
    same parameters (``x <= threshold`` goes left). Then each value of each categorical feature is
    tested (``x == value`` goes left), and the best categorical split replaces the numerical one if
    its gain is higher.

    Parameters
    ----------
    criterion : {'gini', 'entropy'} or callable, default='gini'
        Impurity measure. A callable ``f(y, sample_weight, class_weight) -> float`` is supported only
        for categorical features.
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
    class_weight : dict, 'balanced' or None, default=None
        Weights of the classes in the impurity; 'balanced' weighs them inversely to their frequency.

    Attributes
    ----------
    feature_ : int
        Column of the feature matrix used by the split.
    threshold_ : float or object
        Threshold of a numerical split, or value of a categorical split.
    is_categorical_ : bool
        Whether the split is categorical.
    gain_ : float
        Information gain of the split.
    impurity_ : list of float
        Impurity of the node, of the left child and of the right child.

    See Also
    --------
    RegressionSplitter : the same splitter for regression.
    DecisionTreeStumpClassifier : the stump made of a MatrixTransformer and this splitter.

    Notes
    -----
    The categorical splits do not use ``sample_weight`` and ignore ``min_samples_leaf``.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from RuleTree.stumps.splitters import ClassificationSplitter
    >>> X, y = load_iris(return_X_y=True)
    >>> splitter = ClassificationSplitter(random_state=0).fit(X, y, numerical=[0, 1, 2, 3], categorical=[])
    >>> splitter.feature_, round(float(splitter.threshold_), 2), round(float(splitter.gain_), 3)
    (3, 0.8, 0.333)
    >>> splitter.apply(X[[0, 100]])
    array([1, 2])
    """

    task = MODEL_TYPE_CLF

    def __init__(self, criterion='gini', min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                 random_state=None, min_impurity_decrease=0.0, class_weight=None):
        """Initialize the ClassificationSplitter; see the class documentation for the parameters."""
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight

    def _impurity_fun(self):
        """
        Impurity function of the criterion.

        Returns
        -------
        impurity_fun : callable
            Function (y, sample_weight, class_weight) -> impurity.
        """
        if self.criterion == "gini":
            return gini
        elif self.criterion == "entropy":
            return entropy
        return self.criterion

    def _class_weight(self, y):
        """
        Weight of each class for class_weight='balanced'.

        Parameters
        ----------
        y : numpy.ndarray
            Target of the instances in the node.

        Returns
        -------
        class_weight : dict or None
            Class -> weight, or None if class_weight is not 'balanced'.
        """
        if self.class_weight != "balanced":
            return None

        classes = np.unique(y)
        return {class_label: len(y) / (len(classes) * np.sum(y == class_label)) for class_label in classes}

    def fit(self, X, y, numerical, categorical, sample_weight=None, check_input=True):
        """
        Find the best split: the threshold of the scikit-learn tree on the numerical features, replaced by a
        categorical split if one has a higher information gain. On a binary feature, the threshold is halfway
        between its two values.

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
        self : ClassificationSplitter
            The fitted splitter.

        Raises
        ------
        NoSplitFoundWarning
            If no split is found.
        """
        impurity_fun = self._impurity_fun()
        class_weight = self._class_weight(y)
        self.feature_, self.threshold_, self.is_categorical_, self.impurity_ = None, None, False, None
        best_info_gain = -float('inf')

        if len(numerical) > 0:
            tree = DecisionTreeClassifier(max_depth=1, criterion=self.criterion, min_samples_leaf=self.min_samples_leaf,
                                          min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                          max_features=self.max_features, random_state=self.random_state,
                                          min_impurity_decrease=self.min_impurity_decrease,
                                          class_weight=self.class_weight)
            tree.fit(X[:, numerical], y, sample_weight=sample_weight, check_input=check_input)

            if tree.tree_.node_count == 1:
                raise NoSplitFoundWarning(f"No split found for X {X.shape} and y {np.unique(y)}")

            self.feature_ = numerical[tree.tree_.feature[0]]
            self.threshold_ = tree.tree_.threshold[0]
            best_info_gain = get_info_gain(tree)

            X_feature = X[:, self.feature_]
            self.impurity_ = [
                impurity_fun(y, sample_weight, class_weight),
                impurity_fun(y[X_feature <= self.threshold_], None, class_weight),
                impurity_fun(y[X_feature > self.threshold_], None, class_weight)
            ]

        best_info_gain = self._fit_categorical(X, y, categorical, best_info_gain, impurity_fun, class_weight)

        if self.feature_ is None:
            raise NoSplitFoundWarning(f"No split found for X {X.shape} and y {np.unique(y)}")

        X_feature = X[:, self.feature_]
        if not self.is_categorical_ and len(np.unique(X_feature)) == 2:
            # binary feature: threshold halfway between the two values
            self.threshold_ = (X_feature.max() + X_feature.min()) / 2

        self.gain_ = best_info_gain

        return self

    def _fit_categorical(self, X, y, categorical, best_info_gain, impurity_fun, class_weight):
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
        impurity_fun : callable
            Impurity function.
        class_weight : dict or None
            Weight of each class.

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

                info_gain = _get_info_gain(impurity_fun(y, None, class_weight),
                                           impurity_fun(y[X_split], None, class_weight),
                                           impurity_fun(y[~X_split], None, class_weight),
                                           len_x,
                                           len_left,
                                           len_x - len_left)

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    self.feature_ = i
                    self.threshold_ = value
                    self.is_categorical_ = True
                    self.impurity_ = [
                        impurity_fun(y, None, class_weight),
                        impurity_fun(y[X_split], None, class_weight),
                        impurity_fun(y[~X_split], None, class_weight)
                    ]

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
        impurity_fun = self._impurity_fun()
        class_weight = self._class_weight(y)

        left = self.apply(X) == 1
        self.impurity_ = [
            impurity_fun(y, sample_weight, class_weight),
            impurity_fun(y[left], None, class_weight),
            impurity_fun(y[~left], None, class_weight)
        ]
