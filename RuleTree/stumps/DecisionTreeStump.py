"""DecisionTreeStump: what DecisionTreeStumpClassifier and DecisionTreeStumpRegressor have in common."""

from RuleTree.stumps.Stump import Stump
from RuleTree.stumps.transformers.MatrixTransformer import MatrixTransformer


class DecisionTreeStump(Stump):
    """
    Base class of the decision tree stumps (DecisionTreeStumpClassifier and DecisionTreeStumpRegressor):
    a MatrixTransformer and the splitter ``_splitter_class`` (ClassificationSplitter or
    RegressionSplitter) with the parameters of the stump.

    A subclass defines its parameters (``__init__``) and ``_splitter_class``.
    """

    # splitter of the stump, built with the parameters of _splitter_params
    _splitter_class = None

    def _splitter_params(self) -> dict:
        """
        Parameters of the splitter (also the parameters of the trees of the pivot and oblique
        transformers).

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
        A new, unfitted splitter of class ``_splitter_class`` with the parameters of the stump.

        Returns
        -------
        splitter : DecisionTreeSplitter
        """
        return self._splitter_class(**self._splitter_params())
