"""RegressionSplitter: the best split for regression."""

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

from RuleTree.stumps.splitters.DecisionTreeSplitter import DecisionTreeSplitter
from RuleTree.utils.define import MODEL_TYPE_REG
from RuleTree.utils.impurity import regression_impurity, regression_statistics


class RegressionSplitter(DecisionTreeSplitter):
    """
    Splitter for regression that maximizes the decrease of impurity.

    The numerical features are split by a scikit-learn ``DecisionTreeRegressor`` of depth 1 with the
    same parameters (``x <= threshold`` goes left). Then each value of each categorical feature is
    tested (``x == value`` goes left), and the best categorical split replaces the numerical one if
    its gain is higher (see DecisionTreeSplitter).

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
    On a binary feature, the threshold is halfway between its two values; the categorical values that
    would leave a child empty are not tested.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from RuleTree.stumps.splitters import RegressionSplitter
    >>> X, y = load_diabetes(return_X_y=True)
    >>> splitter = RegressionSplitter(random_state=0)
    >>> splitter = splitter.fit(X, y, numerical=list(range(10)), categorical=[], context=None, node_id=None, cache=None)
    >>> splitter.feature_, round(float(splitter.threshold_), 4)
    (8, -0.0038)
    """

    task = MODEL_TYPE_REG

    _sklearn_tree = DecisionTreeRegressor

    def __init__(self, criterion='squared_error', min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features=None, random_state=None, min_impurity_decrease=0.0):
        """Initialize the RegressionSplitter; see the class documentation for the parameters."""
        super().__init__(criterion, min_samples_leaf, min_weight_fraction_leaf, max_features, random_state,
                         min_impurity_decrease)

    def _impurities(self, y, left, sample_weight=None):
        """Impurity of the node and of the two children (see :meth:`_impurity`)."""
        return [self._impurity(y), self._impurity(y[left]), self._impurity(y[~left])]

    def _impurity(self, y) -> float:
        """
        Impurity of a node, 0 if empty: with several targets, the average over the targets. With
        'absolute_error' or a callable, ``f(y_true, y_pred)`` with each target predicted with its mean.
        """
        if len(y) == 0:
            return 0
        if self.criterion == "absolute_error" or callable(self.criterion):
            impurity = mean_absolute_error if self.criterion == "absolute_error" else self.criterion
            return impurity(y_true=y, y_pred=np.broadcast_to(np.mean(y, axis=0), y.shape))
        return float(np.mean([regression_impurity(regression_statistics(target, self.criterion).sum(axis=0),
                                                  self.criterion) for target in y.reshape(len(y), -1).T]))
