"""PivotTreeStump: what the pivot stumps for classification and regression have in common."""

from RuleTree.stumps.transformers.PivotTransformer import PivotTransformer


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
        """The pivot of the split (standardized numerical features), None if not fitted."""
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
        >>> pivots = PivotTreeStumpClassifier(random_state=0).fit(X, y).get_pivots()
        >>> pivots["descriptives"], pivots["discriminatives"], pivots["used"]
        ([7, 96, 112], [27, 77, 127], 112)
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

        feat_name = f"P_{self.feature_original}"
        threshold = self.threshold_original
        rounded_value = str(threshold) if float_precision is None else round(threshold, float_precision)
        condition, negation = f"{feat_name} <= {rounded_value}", f"{feat_name} > {rounded_value}"

        return {
            "feature_idx": self.feature_original,
            "threshold": threshold,
            "is_categorical": False,
            "feature_name": feat_name,
            "textual_rule": condition,
            "blob_rule": condition,
            "graphviz_rule": {"label": condition},
            "not_textual_rule": negation,
            "not_blob_rule": negation,
            "not_graphviz_rule": {"label": negation},
        }
