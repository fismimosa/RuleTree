"""FairTreeStumpRegressor: the split for regression under a fairness constraint."""

from typing import Union

import psutil

from RuleTree.stumps.regression.DecisionTreeStumpRegressor import DecisionTreeStumpRegressor
from RuleTree.stumps.splitters.FairSplitter import FairSplitter
from RuleTree.utils.define import DATA_TYPE_TABULAR


class FairTreeStumpRegressor(DecisionTreeStumpRegressor):
    """
    Fair stump for regression: the threshold on a numerical feature with the highest gain, penalized
    by how unfair the split is with respect to a sensitive attribute.

    The sensitive attribute is a column of the data (``sensible_attribute``), never used by the
    splits. The fairness criterion (``penalty``) is statistical parity of the children, a maximum
    fairness cost with respect to an ideal distribution, or privacy constraints (k-anonymity,
    l-diversity, t-closeness). It is a :class:`~RuleTree.stumps.Stump.Stump` made of a
    MatrixTransformer and a FairSplitter; it supports only tabular data.

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
    n_jobs : int, default=number of physical CPUs
        Number of processes used to evaluate the features.
    criterion : {'squared_error', 'absolute_error', 'poisson'} or callable, default='squared_error'
        Impurity measure (see RegressionSplitter).

    Attributes
    ----------
    transformer_ : MatrixTransformer
        The fitted transformer.
    splitter_ : FairSplitter
        The fitted splitter (feature, threshold, gain and impurity of the split).
    data_type_ : str
        Data type the stump was fitted on.
    feature_original : int
        Column of the data used by the split.
    threshold_original : float
        Threshold of the split.

    See Also
    --------
    FairSplitter : the splitter, usable with other transformers (e.g. pivots).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_diabetes
    >>> from RuleTree.stumps.regression import FairTreeStumpRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> X = np.column_stack([X, X[:, 1] > 0])  # last column: the sensitive attribute
    >>> stump = FairTreeStumpRegressor(penalty="balance", sensible_attribute=-1, n_jobs=1).fit(X, y)
    >>> stump.get_rule()["textual_rule"]
    'X_4 <= -0.03'
    """

    # sensible_attribute is a column of the tabular data
    supported_data_types = (DATA_TYPE_TABULAR,)

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
                 n_jobs: int = psutil.cpu_count(logical=False),
                 criterion='squared_error'):
        """Initialize the FairTreeStumpRegressor; see the class documentation for the parameters."""
        self.penalty = penalty
        self.sensible_attribute = sensible_attribute
        self.penalization_weight = penalization_weight
        self.ideal_distribution = ideal_distribution
        self.k_anonymity = k_anonymity
        self.l_diversity = l_diversity
        self.t_closeness = t_closeness
        self.strict = strict
        self.use_t = use_t
        self.n_jobs = n_jobs
        self.criterion = criterion

    def _make_splitter(self):
        """
        A new, unfitted FairSplitter with the parameters of the stump.

        Returns
        -------
        splitter : FairSplitter
        """
        return FairSplitter(penalty=self.penalty, sensible_attribute=self.sensible_attribute,
                            penalization_weight=self.penalization_weight, ideal_distribution=self.ideal_distribution,
                            k_anonymity=self.k_anonymity, l_diversity=self.l_diversity, t_closeness=self.t_closeness,
                            strict=self.strict, use_t=self.use_t, n_jobs=self.n_jobs, criterion=self.criterion)
