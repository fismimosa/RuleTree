"""ObliqueTreeStump: what the oblique stumps for classification and regression have in common."""

from RuleTree.base.RuleTreeBaseStump import numerical_rule
from RuleTree.stumps.transformers.BivariateTransformer import BivariateTransformer
from RuleTree.stumps.transformers.HouseholderTransformer import HouseholderTransformer

OBLIQUE_SPLIT_TYPES = ("householder", "bivariate")


def check_oblique_split_type(oblique_split_type: str):
    """
    Check the type of oblique split of a stump.

    Parameters
    ----------
    oblique_split_type : str
        'householder' or 'bivariate'.

    Raises
    ------
    ValueError
        For any other value.
    """
    if oblique_split_type not in OBLIQUE_SPLIT_TYPES:
        raise ValueError(f"'oblique_split_type' must be one of {OBLIQUE_SPLIT_TYPES}, got {oblique_split_type!r}")


def format_combination(names: list, coefficients: list, float_precision=3) -> str:
    """
    Text of a linear combination, e.g. ``0.8 * X_2 - 0.6 * X_3``; a single term with coefficient 1
    (an axis-parallel direction) is just its name.

    Parameters
    ----------
    names : list of str
        Names of the terms.
    coefficients : list of float
        Their coefficients.
    float_precision : int, default=3
        Number of decimals of the coefficients. If None, they are not rounded.

    Returns
    -------
    text : str

    Examples
    --------
    >>> from RuleTree.stumps.ObliqueTreeStump import format_combination
    >>> format_combination(["X_0", "X_3"], [0.5, -0.25])
    '0.5 * X_0 - 0.25 * X_3'
    >>> format_combination(["X_3"], [1.0])
    'X_3'
    """
    if len(names) == 1 and coefficients[0] == 1:
        return names[0]

    text = ""
    for i, (name, coefficient) in enumerate(zip(names, coefficients)):
        value = coefficient if float_precision is None else round(coefficient, float_precision)
        if i == 0:
            text = f"{value} * {name}"
        else:
            text += f" {'-' if value < 0 else '+'} {abs(value)} * {name}"
    return text


def combination_rule(feature_idx, names, coefficients, threshold, float_precision=3) -> dict:
    """
    Rule of a threshold on a linear combination: ``combination <= threshold`` goes left.

    Parameters
    ----------
    feature_idx : object
        Identifier of the combination, stored in ``feature_idx``.
    names : list of str
        Names of the terms.
    coefficients : list of float
        Their coefficients.
    threshold : float
        Threshold of the split.
    float_precision : int, default=3
        Number of decimals of the coefficients and of the threshold. If None, they are not rounded.

    Returns
    -------
    rule : dict
        See RuleTreeBaseStump.get_rule; also ``coefficients``.
    """
    return numerical_rule(feature_idx, format_combination(names, coefficients, float_precision), threshold,
                          float_precision, coefficients=coefficients)


class ObliqueTreeStump:
    """
    Mixin of the oblique stumps (ObliqueDecisionTreeStumpClassifier and
    ObliqueDecisionTreeStumpRegressor): the oblique transformer built from the parameters of the
    stump, and the rule of the split.

    The subclass defines the parameters and the splitter, by deriving also from the decision tree
    stump of its task (``class ObliqueDecisionTreeStumpClassifier(ObliqueTreeStump, DecisionTreeStumpClassifier)``).
    """

    def _make_transformer(self):
        """
        A new, unfitted HouseholderTransformer or BivariateTransformer, according to
        ``oblique_split_type``; the trees of the bivariate transformer use the parameters of the
        splitter.

        Returns
        -------
        transformer : ObliqueTransformer
        """
        if self.oblique_split_type == "householder":
            return HouseholderTransformer(max_oblique_features=self.max_oblique_features, tau=self.tau,
                                          random_state=self.random_state)
        return BivariateTransformer(ml_task=self.task, n_orientations=self.n_orientations,
                                    tree_params=self._splitter_params())

    @property
    def coefficients(self):
        """Coefficients of the combination of the split, None if not fitted."""
        return self.transformer_.combination(self.splitter_.feature_)[1] if self._is_fitted() else None

    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        """
        Rule of the split, e.g. ``0.8 * X_2 - 0.6 * X_3 <= 1.5``.

        Parameters
        ----------
        columns_names : list of str, default=None
            Names of the columns of the data, used instead of ``X_<index>``.
        scaler : TransformerMixin, default=None
            Not supported: must be None.
        float_precision : int, default=3
            Number of decimals of the coefficients and of the threshold; None for no rounding.

        Returns
        -------
        rule : dict
            ``feature_idx`` (the columns of the combination), ``coefficients``, ``threshold``,
            ``is_categorical``, ``feature_name`` (the combination), the condition of the left child
            (``textual_rule``, ``blob_rule``, ``graphviz_rule``) and its negation. Empty if the
            stump is not fitted.

        Raises
        ------
        ValueError
            If a scaler is given.
        """
        if scaler is not None:
            raise ValueError(f"Scaler not supported for {self.__class__.__name__}")
        if not self._is_fitted():
            return {}

        features, coefficients = self.transformer_.combination(self.splitter_.feature_)
        names = [f"X_{f}" if columns_names is None else columns_names[f] for f in features]
        return combination_rule(features, names, coefficients, self.threshold_original, float_precision)
