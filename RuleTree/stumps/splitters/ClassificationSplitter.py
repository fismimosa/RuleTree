"""ClassificationSplitter: the best split for classification."""

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from RuleTree.stumps.splitters.DecisionTreeSplitter import DecisionTreeSplitter
from RuleTree.utils.data_utils import entropy, gini
from RuleTree.utils.define import MODEL_TYPE_CLF


class ClassificationSplitter(DecisionTreeSplitter):
    """
    Splitter for classification that maximizes the decrease of impurity (information gain).

    The numerical features are split by a scikit-learn ``DecisionTreeClassifier`` of depth 1 with the
    same parameters (``x <= threshold`` goes left). Then each value of each categorical feature is
    tested (``x == value`` goes left), and the best categorical split replaces the numerical one if
    its gain is higher (see DecisionTreeSplitter).

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
    On a binary feature, the threshold is halfway between its two values; the categorical values that
    would leave a child empty are not tested.

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

    _sklearn_tree = DecisionTreeClassifier

    def __init__(self, criterion='gini', min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                 random_state=None, min_impurity_decrease=0.0, class_weight=None):
        """Initialize the ClassificationSplitter; see the class documentation for the parameters."""
        super().__init__(criterion, min_samples_leaf, min_weight_fraction_leaf, max_features, random_state,
                         min_impurity_decrease)
        self.class_weight = class_weight

    def _tree_params(self):
        """
        Parameters of the DecisionTreeClassifier besides those of DecisionTreeSplitter: class_weight.

        Returns
        -------
        params : dict
        """
        return {"class_weight": self.class_weight}

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

    def _impurities(self, y, left, sample_weight=None):
        """
        Impurity of the node (with the sample weights) and of the two children, weighing the classes
        as class_weight.

        Parameters
        ----------
        y : numpy.ndarray of shape (n_samples,)
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
        impurity_fun = self._impurity_fun()
        class_weight = self._class_weight(y)
        return [impurity_fun(y, sample_weight, class_weight),
                impurity_fun(y[left], None, class_weight),
                impurity_fun(y[~left], None, class_weight)]
