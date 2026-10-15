"""XGBoostRegressor: gradient boosting with the trees of XGBoost."""

from psutil import cpu_count

from RuleTree.ensemble.GBoostRegressor import GBoostRegressor
from RuleTree.stumps.regression.XGBoostStumpRegressor import XGBoostStumpRegressor
from RuleTree.tree.RuleTreeXGBoostRegressor import RuleTreeXGBoostRegressor


class XGBoostRegressor(GBoostRegressor):
    """
    Gradient boosting for regression with the trees of XGBoost (RuleTreeXGBoostRegressor): the
    splits maximize the XGBoost gain and the leaves predict the regularized XGBoost values.

    Parameters
    ----------
    base_estimator : RuleTreeXGBoostRegressor, default=None
        Tree fitted at each iteration. If None, RuleTreeXGBoostRegressor with the XGBoost stumps and
        as many jobs as CPUs, built at fit time.
    learning_rate : float, default=0.3
        Shrinkage of the contribution of each tree.
    n_estimators, loss, patience, val_size, random_state
        As in GBoostRegressor.

    Attributes
    ----------
    base_prediction_ : float
        Initial prediction (mean of the target).
    estimators_ : list of RuleTreeXGBoostRegressor
        Fitted trees.

    See Also
    --------
    RuleTree.tree.RuleTreeXGBoostRegressor.RuleTreeXGBoostRegressor : the base tree.
    GBoostRegressor : the same boosting with any RuleTreeRegressor.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from RuleTree.ensemble.XGBoostRegressor import XGBoostRegressor
    >>> from RuleTree.stumps.regression import XGBoostStumpRegressor
    >>> from RuleTree.tree.RuleTreeXGBoostRegressor import RuleTreeXGBoostRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> tree = RuleTreeXGBoostRegressor(max_depth=2, lam=1, base_stumps=XGBoostStumpRegressor(lam=1))
    >>> model = XGBoostRegressor(base_estimator=tree, n_estimators=10).fit(X, y)
    >>> round(model.score(X, y), 3)
    0.569
    """
    def __init__(self, base_estimator=None, n_estimators=100, learning_rate=.3, loss='squared_loss', patience=5,
                 val_size=.15, random_state=42):
        """Initialize the XGBoostRegressor; see the class documentation for the parameters."""
        # base_estimator=None: the default RuleTreeXGBoostRegressor is built in fit, not at import time
        # explicit parameters (not **kwargs): get_params and clone see only the ones in the signature
        super().__init__(base_estimator=base_estimator, n_estimators=n_estimators, learning_rate=learning_rate,
                         loss=loss, patience=patience, val_size=val_size, random_state=random_state)

    def _get_base_estimator(self):
        """
        Tree copied at each iteration: base_estimator, or the default RuleTreeXGBoostRegressor.

        Returns
        -------
        base_estimator : RuleTreeXGBoostRegressor
        """
        if self.base_estimator is not None:
            return self.base_estimator

        n_jobs = cpu_count()
        return RuleTreeXGBoostRegressor(n_jobs=n_jobs, base_stumps=XGBoostStumpRegressor(lam=1, n_jobs=n_jobs))

    def _get_base_prediction(self, y):
        """
        Initial prediction of the boosting.

        Parameters
        ----------
        y : numpy.ndarray
            Target of the training set.

        Returns
        -------
        base_prediction : float
            0.5, as in XGBoost.
        """
        return .5
