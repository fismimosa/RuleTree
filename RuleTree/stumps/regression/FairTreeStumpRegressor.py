"""FairTreeStumpRegressor: the split for regression under a fairness constraint."""

from typing import Union

import psutil

from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from RuleTree.stumps.regression.DecisionTreeStumpRegressor import DecisionTreeStumpRegressor
from RuleTree.stumps.splitters.FairSplitter import FairSplitter
from RuleTree.utils.define import DATA_TYPE_TABULAR


class FairTreeStumpRegressor(DecisionTreeStumpRegressor):
    """
    Fair Decision Tree Stump Regressor that enforces fairness constraints during training.

    This class extends DecisionTreeStumpRegressor by introducing fairness constraints during the
    learning process. It can enforce balancing, maximum fairness cost, or privacy-related
    constraints (k-anonymity, l-diversity, t-closeness) with respect to a sensitive attribute.

    Parameters
    ----------
    penalty : str, default=None
        Type of fairness constraint to apply. Options are:
        - "balance": Enforce statistical parity (equal representation)
        - "mfc": Maximum Fairness Cost according to ideal distribution
        - "privacy": Enforce privacy constraints (k-anonymity, l-diversity, t-closeness)
        - None: No fairness constraints (equivalent to standard DecisionTreeStumpRegressor)

    sensible_attribute : int, default=-1
        The index of the column in X containing the sensitive/protected attribute.

    penalization_weight : float, default=0.3
        Weight of the fairness penalty term. Higher values enforce stronger fairness.

    ideal_distribution : dict, default=None
        Required when penalty="mfc". Dictionary specifying the ideal distribution for different groups.

    k_anonymity : int or float, default=2
        Minimum group size for k-anonymity (used when penalty="privacy").

    l_diversity : int or float, default=2
        Minimum distinct values required for l-diversity (used when penalty="privacy").

    t_closeness : float, default=0.2
        Maximum earth mover's distance allowed for t-closeness (used when penalty="privacy").

    strict : bool, default=True
        If True, strictly enforces privacy constraints; if False, uses penalization instead.

    use_t : bool, default=True
        Whether to use t-closeness in privacy constraint calculations.

    n_jobs : int, default=psutil.cpu_count(logical=False)
        Number of CPU cores to use for parallel processing.

    criterion : str or callable, default='squared_error'
        Impurity measure (see RegressionSplitter).
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

    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        """
        Get a human-readable rule representation of the tree stump.

        Parameters
        ----------
        columns_names : array-like, default=None
            Names of the features. If None, generic feature names are used.

        scaler : object, default=None
            If not None, the scaler is used to transform the threshold values
            back to the original scale.

        float_precision : int, default=3
            Number of decimal places to use when rounding threshold values.

        Returns
        -------
        rule : dict
            Dictionary representation of the learned rule.
        """
        return DecisionTreeStumpClassifier.get_rule(self,
                                                    columns_names=columns_names,
                                                    scaler=scaler,
                                                    float_precision=float_precision)
