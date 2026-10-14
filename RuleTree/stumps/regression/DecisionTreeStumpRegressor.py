"""DecisionTreeStumpRegressor: the split on one feature for regression."""

from RuleTree.stumps.Stump import Stump
from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from RuleTree.stumps.splitters.RegressionSplitter import RegressionSplitter
from RuleTree.stumps.transformers.MatrixTransformer import MatrixTransformer


class DecisionTreeStumpRegressor(Stump):
    """
    A decision tree stump for regression: a single split on one feature.

    Numerical features are split with a threshold (``x <= t``), categorical features on a value
    (``x == v``): the split with the highest decrease of impurity is selected. Time series and
    images are flattened into one feature per value.

    It is made of a MatrixTransformer and a RegressionSplitter.

    Parameters
    ----------
    criterion : {'squared_error', 'absolute_error', 'poisson'}, default='squared_error'
        Impurity measure.
    min_samples_leaf, min_weight_fraction_leaf, max_features, random_state, min_impurity_decrease
        As in sklearn.tree.DecisionTreeRegressor, for the numerical features. With random_state=None,
        ties between features with the same gain are broken at random.
    """

    def __init__(self, criterion='squared_error', min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features=None, random_state=None, min_impurity_decrease=0.0):
        """Initialize the DecisionTreeStumpRegressor; see the class documentation for the parameters."""
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease

    def _splitter_params(self) -> dict:
        """
        Parameters of the RegressionSplitter.

        Returns
        -------
        params : dict
        """
        return dict(criterion=self.criterion, min_samples_leaf=self.min_samples_leaf,
                    min_weight_fraction_leaf=self.min_weight_fraction_leaf, max_features=self.max_features,
                    random_state=self.random_state, min_impurity_decrease=self.min_impurity_decrease)

    def _make_transformer(self):
        """
        A new MatrixTransformer: the data are used as they are (time series and images flattened).

        Returns
        -------
        transformer : MatrixTransformer
        """
        return MatrixTransformer()

    def _make_splitter(self):
        """
        A new, unfitted RegressionSplitter with the parameters of the stump.

        Returns
        -------
        splitter : RegressionSplitter
        """
        return RegressionSplitter(**self._splitter_params())

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
