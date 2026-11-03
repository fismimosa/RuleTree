"""
PivotTreeStump, ProximityPivotTreeStump and ObliquePivotTreeStump: what the pivot stumps have in
common.
"""

from RuleTree.base.RuleTreeBaseStump import numerical_rule
from RuleTree.stumps.ObliqueTreeStump import combination_rule
from RuleTree.stumps.ProximityTreeStump import proximity_rule
from RuleTree.stumps.transformers.ObliquePivotTransformer import ObliquePivotTransformer
from RuleTree.stumps.transformers.PivotTransformer import PivotTransformer
from RuleTree.stumps.transformers.ProximityPivotTransformer import ProximityPivotTransformer
from RuleTree.utils.deprecation import deprecated_aliases


class PivotTreeStump:
    """
    Mixin of the pivot stumps (PivotTreeStumpClassifier and PivotTreeStumpRegressor): the
    PivotTransformer built from the parameters of the stump, the pivots of the split and its rule.

    The subclass defines the parameters and the splitter, by deriving also from the decision tree
    stump of its task (``class PivotTreeStumpClassifier(PivotTreeStump, DecisionTreeStumpClassifier)``).
    """

    def _make_transformer(self):
        """
        A new, unfitted PivotTransformer with the parameters of the stump; its decision trees use the
        parameters of the splitter.

        Returns
        -------
        transformer : PivotTransformer
        """
        return PivotTransformer(ml_task=self.task, distance_measure=self.distance_measure, k=self.k,
                                reg_contrast_lambda=self.reg_contrast_lambda,
                                reg_max_thresholds=self.reg_max_thresholds, tree_params=self._splitter_params())

    @property
    def X_split_instance(self):
        """The pivot of the split, standardized (categorical features one-hot encoded); None if not fitted."""
        return self.transformer_.candidates_[0] if self._is_fitted() else None

    def get_pivots(self):
        """
        Pivots of the node, as indices of instances of the training set.

        Returns
        -------
        pivots : dict
            ``descriptives`` (medoids), ``discriminatives`` and ``candidates`` (both) pivots of the
            node, and the pivot ``used`` by the split.

        Examples
        --------
        >>> from sklearn.datasets import load_iris
        >>> from RuleTree.stumps.classification import PivotTreeStumpClassifier
        >>> X, y = load_iris(return_X_y=True)
        >>> pivots = PivotTreeStumpClassifier(random_state=0)
        >>> pivots = pivots.fit(X, y, context=None, node_id=None, cache=None).get_pivots()
        >>> pivots["descriptives"], pivots["discriminatives"], pivots["used"]
        ([7, 99, 147], [27, 76, 108], 76)
        """
        return self.transformer_.pivots_ | {"used": int(self.feature_original)}

    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        """
        Rule of the split, ``P_<i> <= threshold``, where ``P_<i>`` is the distance from the training
        instance ``i``.

        Parameters
        ----------
        columns_names : list of str, default=None
            Not used: the rule does not name the features.
        scaler : TransformerMixin, default=None
            Not used: the distances are computed on the data standardized by the stump.
        float_precision : int, default=3
            Number of decimals of the threshold; None for no rounding.

        Returns
        -------
        rule : dict
            ``feature_idx`` (the pivot), ``threshold``, ``is_categorical``, ``feature_name``, the
            condition of the left child (``textual_rule``, ``blob_rule``, ``graphviz_rule``) and its
            negation (``not_textual_rule``, ``not_blob_rule``, ``not_graphviz_rule``). Empty if the
            stump is not fitted.
        """
        if not self._is_fitted():
            return {}

        return numerical_rule(self.feature_original, f"P_{self.feature_original}", self.threshold_original,
                              float_precision)

    def get_plot_rule(self, columns_names=None, scaler=None, float_precision=3):
        """
        What the node of the split shows in ``RuleTree.plot_tree``: the rule, and the pivot, drawn below it
        with ``show_pivots``.

        Parameters
        ----------
        columns_names : list of str, default=None
            Not used.
        scaler : TransformerMixin, default=None
            Not used.
        float_precision : int, default=3
            Number of decimals of the threshold.

        Returns
        -------
        rule : dict
            ``label`` and ``pivots`` (the index of the pivot in the training set).
        """
        return {"label": self.get_rule(float_precision=float_precision)["textual_rule"],
                "pivots": [int(self.feature_original)]}


class ProximityPivotTreeStump:
    """
    Mixin of the proximity pivot stumps (ProximityPivotTreeStumpClassifier and
    ObliqueProximityPivotTreeStumpClassifier): the ProximityPivotTransformer built from the parameters
    of the stump, the pivots of the split and its rule, ``P_<a> <= P_<b>`` (closer to the training
    instance ``a`` than to ``b``).
    """

    def _selection_params(self) -> dict:
        """Parameters of the selection of the discriminative pivots; the default ones of a decision tree."""
        return {}

    def _make_transformer(self):
        """
        A new, unfitted ProximityPivotTransformer with the parameters of the stump; its decision trees use
        the parameters of the splitter.

        Returns
        -------
        transformer : ProximityPivotTransformer
        """
        return ProximityPivotTransformer(ml_task=self.task, distance_measure=self.distance_measure,
                                        tree_params=self._splitter_params(), **self._selection_params())

    def get_pivots(self):
        """
        Pivots of the node, as indices of instances of the training set.

        Returns
        -------
        pivots : dict
            ``descriptives``, ``discriminatives`` and ``candidates`` pivots of the node, and the two
            pivots ``used`` by the split.
        """
        return self.transformer_.pivots_ | {"used": list(self.feature_original)}

    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        """
        Rule of the split, ``P_<a> <= P_<b>``: the instances closer to the training instance ``a`` than to
        ``b`` go left.

        Parameters
        ----------
        columns_names : list of str, default=None
            Not used: the rule does not name the features.
        scaler : TransformerMixin, default=None
            Not used: the distances are computed on the data standardized by the stump.
        float_precision : int, default=3
            Not used: the rule has no number.

        Returns
        -------
        rule : dict
            See RuleTreeBaseStump.get_rule. Empty if the stump is not fitted.
        """
        if not self._is_fitted():
            return {}
        first, second = self.feature_original
        return proximity_rule(self.feature_original, f"P_{first}", f"P_{second}", self.threshold_original)

    def get_plot_rule(self, columns_names=None, scaler=None, float_precision=3):
        """
        What the node of the split shows in ``RuleTree.plot_tree``: the rule, and the two pivots, drawn
        with ``show_pivots`` on the arrows of the children: the first one on the left arrow ("true", the
        instances closer to it), the second one on the right arrow.

        Parameters
        ----------
        columns_names : list of str, default=None
            Not used.
        scaler : TransformerMixin, default=None
            Not used.
        float_precision : int, default=3
            Not used.

        Returns
        -------
        rule : dict
            ``label``, ``pivots`` (the indices of the two pivots in the training set) and
            ``pivots_on_edges`` (True).
        """
        return {"label": self.get_rule()["textual_rule"], "pivots": [int(p) for p in self.feature_original],
                "pivots_on_edges": True}


class ObliquePivotTreeStump:
    """
    Mixin of the oblique pivot stumps (ObliquePivotTreeStumpClassifier): the ObliquePivotTransformer
    built from the parameters of the stump, the pivots of the split and its rule, e.g.
    ``0.7 * P_12 - 0.3 * P_57 <= 1.2``.
    """

    def _make_transformer(self):
        """
        A new, unfitted ObliquePivotTransformer with the parameters of the stump; its decision trees use
        the parameters of the splitter.

        Returns
        -------
        transformer : ObliquePivotTransformer
        """
        return ObliquePivotTransformer(ml_task=self.task, distance_measure=self.distance_measure,
                                       tree_params=self._splitter_params(),
                                       oblique_split_type=self.oblique_split_type,
                                       max_oblique_features=self.max_oblique_features, tau=self.tau,
                                       n_orientations=self.n_orientations)

    @property
    def coefficients(self):
        """Coefficients of the combination of the split, None if not fitted."""
        return self.transformer_.combination(self.splitter_.feature_)[1] if self._is_fitted() else None

    def get_pivots(self):
        """
        Pivots of the node, as indices of instances of the training set.

        Returns
        -------
        pivots : dict
            ``descriptives``, ``discriminatives`` and ``candidates`` pivots of the node, and the pivots
            ``used`` by the combination of the split.
        """
        return self.transformer_.pivots_ | {"used": list(self.feature_original)}

    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        """
        Rule of the split, e.g. ``0.7 * P_12 - 0.3 * P_57 <= 1.2``, where ``P_<i>`` is the distance from
        the training instance ``i``.

        Parameters
        ----------
        columns_names : list of str, default=None
            Not used: the rule does not name the features.
        scaler : TransformerMixin, default=None
            Not used: the distances are computed on the data standardized by the stump.
        float_precision : int, default=3
            Number of decimals of the coefficients and of the threshold; None for no rounding.

        Returns
        -------
        rule : dict
            See RuleTreeBaseStump.get_rule; also ``coefficients``. Empty if the stump is not fitted.
        """
        if not self._is_fitted():
            return {}
        pivots, coefficients = self.transformer_.combination(self.splitter_.feature_)
        return combination_rule(tuple(pivots), [f"P_{p}" for p in pivots], coefficients, self.threshold_original,
                                float_precision)

    def get_plot_rule(self, columns_names=None, scaler=None, float_precision=3):
        """
        What the node of the split shows in ``RuleTree.plot_tree``: the rule, and the pivots of the
        combination, drawn below it with ``show_pivots``.

        Parameters
        ----------
        columns_names : list of str, default=None
            Not used.
        scaler : TransformerMixin, default=None
            Not used.
        float_precision : int, default=3
            Number of decimals of the coefficients and of the threshold.

        Returns
        -------
        rule : dict
            ``label`` and ``pivots`` (the indices of the pivots in the training set).
        """
        return {"label": self.get_rule(float_precision=float_precision)["textual_rule"],
                "pivots": [int(p) for p in self.feature_original]}


__getattr__ = deprecated_aliases(__name__, {"MultiplePivotTreeStump": ProximityPivotTreeStump})
