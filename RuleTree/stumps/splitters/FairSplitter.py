"""FairSplitter: the best regression split under a fairness constraint."""

from typing import Union

import numpy as np

from RuleTree.stumps.splitters.FlatMatrixRegressionSplitter import FlatMatrixRegressionSplitter
from RuleTree.stumps.splitters.FlatMatrixSplitter import candidate_mask
from RuleTree.utils.fairness_metrics import balance_metric, max_fairness_cost, privacy_metric, privacy_metric_all
from RuleTree.utils.impurity import information_gain


class FairSplitter(FlatMatrixRegressionSplitter):
    """
    Splitter for regression that maximizes the gain penalized by the unfairness of the split with
    respect to a sensitive attribute.

    The candidates are the thresholds of the numerical features, the sensitive attribute excluded, evaluated
    together as in FlatMatrixRegressionSplitter; the gain of a candidate is
    ``sigmoid(gain) * (1 - penalty * penalization_weight)``, where the penalty measures how unfair the
    children are.

    Parameters
    ----------
    penalty : {'balance', 'mfc', 'privacy'}, default=None
        Fairness criterion, which penalizes the gain of the splits:

        - 'balance': the children should have the same distribution of the sensitive attribute
          (statistical parity);
        - 'mfc': maximum fairness cost with respect to ``ideal_distribution``;
        - 'privacy': the children must satisfy k-anonymity, l-diversity and t-closeness on the
          sensitive attribute (with ``strict=True``, the splits that violate them are discarded).
    sensible_attribute : int, default=-1
        Column of the data with the sensitive attribute. It is never used by the splits.
    penalization_weight : float, default=0.3
        Weight of the penalty: the gain of a split is ``sigmoid(gain) * (1 - penalty * weight)``.
    ideal_distribution : dict, default=None
        Value of the sensitive attribute -> ideal fraction, required by penalty='mfc'.
    k_anonymity : int or float, default=2
        Minimum number of instances in each child (penalty='privacy').
    l_diversity : int or float, default=2
        Minimum number of distinct values of the sensitive attribute in each child
        (penalty='privacy').
    t_closeness : float, default=0.2
        Maximum distance between the distribution of the sensitive attribute in a child and in the
        node (penalty='privacy').
    strict : bool, default=True
        Whether the privacy constraints discard the splits that violate them, instead of only
        penalizing them.
    use_t : bool, default=True
        Whether t-closeness is part of the privacy constraints.
    criterion : {'squared_error', 'poisson'}, default='squared_error'
        Impurity measure (see FlatMatrixRegressionSplitter).

    Attributes
    ----------
    feature_ : int
        Column of the feature matrix used by the split.
    threshold_ : float
        Threshold of the split.
    is_categorical_ : bool
        Whether the split is categorical (always False).
    gain_ : float
        Penalized gain of the split.
    impurity_ : list of float
        Impurity of the node, of the left child and of the right child.

    See Also
    --------
    FairTreeStumpRegressor : the stump made of a MatrixTransformer and this splitter.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_diabetes
    >>> from RuleTree import RuleTreeRegressor
    >>> from RuleTree.stumps.Stump import Stump
    >>> from RuleTree.stumps.splitters import FairSplitter
    >>> from RuleTree.stumps.transformers import MatrixTransformer
    >>> X, y = load_diabetes(return_X_y=True)
    >>> X = np.column_stack([X, X[:, 1] > 0])  # last column: the sensitive attribute
    >>> splitter = FairSplitter(penalty="balance", sensible_attribute=-1)
    >>> model = RuleTreeRegressor(max_depth=2, base_stumps=Stump(MatrixTransformer(), splitter)).fit(X, y)
    """

    def __init__(self,
                 penalty: str = None,
                 sensible_attribute: int = -1,
                 penalization_weight: float = 0.3,
                 ideal_distribution: dict = None,
                 k_anonymity: Union[int, float] = 2,
                 l_diversity: Union[int, float] = 2,
                 t_closeness: float = .2,
                 strict: bool = True,
                 use_t: bool = True,
                 criterion='squared_error'):
        """Initialize the FairSplitter; see the class documentation for the parameters."""
        super().__init__(criterion=criterion)
        self.penalty = penalty
        self.sensible_attribute = sensible_attribute
        self.penalization_weight = penalization_weight
        self.ideal_distribution = ideal_distribution
        self.k_anonymity = k_anonymity
        self.l_diversity = l_diversity
        self.t_closeness = t_closeness
        self.strict = strict
        self.use_t = use_t

    def fit(self, X, y, numerical, categorical, sample_weight=None, check_input=True, *, context,
            node_id, cache):
        """
        Find the split with the highest penalized gain among the thresholds of the numerical features (the
        sensitive attribute excluded); see FlatMatrixSplitter.fit for the parameters.

        Returns
        -------
        self : FairSplitter
            The fitted splitter.

        Raises
        ------
        NoSplitFoundWarning
            If no split is found, e.g. all of them violate the privacy constraints.
        ValueError
            If y has several targets.
        """
        sensible_attribute = self.sensible_attribute % X.shape[1]  # also a negative index, e.g. -1
        self._categorical = categorical  # columns compared as categories by t-closeness
        return super().fit(X, y, [feature for feature in numerical if feature != sensible_attribute], [],
                           sample_weight, check_input, context=context, node_id=node_id, cache=cache)

    def _gains(self, impurity, impurity_left, impurity_right, weight_left, weight_right, X, y, features, thresholds,
               is_categorical, sample_weight=None):
        """Information gain squashed by a sigmoid and penalized by the unfairness of each candidate."""
        gains = 1 / (1 + np.exp(-information_gain(impurity, impurity_left, impurity_right, weight_left, weight_right)))
        left = candidate_mask(X, features, thresholds, is_categorical)
        penalties = np.array([self._penalty(X, left[:, candidate]) for candidate in range(left.shape[1])])
        return gains * (1 - penalties * self.penalization_weight)  # nan: a split discarded

    def _penalty(self, X, left) -> float:
        """Unfairness of the split of X into left and right, nan if the split violates the privacy constraints."""
        protected = X[:, self.sensible_attribute]
        if self.penalty == "balance":
            return 1 - balance_metric(left, protected)
        if self.penalty == "mfc":
            return max_fairness_cost(left, protected, self.ideal_distribution)

        can_split, k, l, t = privacy_metric(X, left, self.sensible_attribute, self.k_anonymity, self.l_diversity,
                                            self.t_closeness, self.strict, self._categorical, self.use_t)
        if not can_split:
            return np.nan
        t = t if self.use_t else self.t_closeness
        return float(np.max(privacy_metric_all(k, self.k_anonymity, l, self.l_diversity, t, self.t_closeness)))
