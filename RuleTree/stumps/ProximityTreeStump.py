"""ProximityTreeStump and PartialProximityTreeStump: what the proximity stumps have in common."""

from typing import Optional

from RuleTree.stumps.transformers.PartialProximityTransformer import PartialProximityTransformer
from RuleTree.stumps.transformers.ProximityTransformer import ProximityTransformer
from RuleTree.utils.shapelet_transform.drawing import graphviz_shapelets

# colors of the two shapelets of the split in the Graphviz drawing, as named in the rules
PROXIMITY_COLORS = ("tab:green", "tab:red")


def proximity_rule(feature_idx, first: str, second: str, threshold) -> dict:
    """
    Rule of a proximity split: ``first <= second`` goes left, where first and second are the
    distances from the two references of the split.

    Parameters
    ----------
    feature_idx : object
        Identifier of the pair, stored in ``feature_idx``.
    first, second : str
        Text of the distances from the first and from the second reference.
    threshold : float
        Threshold of the split on the 0/1 comparison (between 0 and 1).

    Returns
    -------
    rule : dict
        See RuleTreeBaseStump.get_rule.
    """
    condition, negation = f"{first} <= {second}", f"{first} > {second}"
    return {
        "feature_idx": feature_idx,
        "threshold": threshold,
        "is_categorical": False,
        "feature_name": f"{first} > {second}",
        "textual_rule": condition,
        "blob_rule": condition,
        "graphviz_rule": {"label": condition},
        "not_textual_rule": negation,
        "not_blob_rule": negation,
        "not_graphviz_rule": {"label": negation},
    }


class ProximityTreeStump:
    """
    Mixin of the proximity stumps for time series (ProximityTreeStumpClassifier and
    ProximityTreeStumpRegressor): the ProximityTransformer built from the parameters of the stump, the
    rule and its drawing.

    The subclass defines the parameters and the splitter, by deriving also from the shapelet stump of
    its task (``class ProximityTreeStumpClassifier(ProximityTreeStump, ShapeletTreeStumpClassifier)``).
    """

    def _make_transformer(self):
        """
        A new, unfitted ProximityTransformer with the parameters of the stump.

        Returns
        -------
        transformer : ProximityTransformer
        """
        return ProximityTransformer(n_shapelets=self.n_shapelets,
                                    n_shapelets_for_selection=self.n_shapelets_for_selection,
                                    n_ts_for_selection_per_class=self.n_ts_for_selection_per_class,
                                    sliding_window=self.sliding_window,
                                    selection=self.selection,
                                    distance=self.distance,
                                    mi_n_neighbors=self.mi_n_neighbors,
                                    random_state=self.random_state,
                                    n_jobs=self.n_jobs,
                                    ml_task=self.task)

    def get_rule(self, columns_names=None, scaler=None, float_precision: Optional[int] = 3):
        """
        Rule of the split, ``euclidean(TS, green_shp) <= euclidean(TS, red_shp)``: the time series closer
        to the first shapelet (green in the drawing, ``st.shapelets[0]``) than to the second one (red,
        ``st.shapelets[1]``) go left.

        Parameters
        ----------
        columns_names : list of str, default=None
            Not used.
        scaler : TransformerMixin, default=None
            Not supported: must be None.
        float_precision : int, default=3
            Not used: the rule has no number.

        Returns
        -------
        rule : dict
            See RuleTreeBaseStump.get_rule.

        Raises
        ------
        ValueError
            If a scaler is given.
        """
        if scaler is not None:
            raise ValueError(f"Scaler not supported for {self.__class__.__name__}")

        return proximity_rule(self.feature_original, f"{self.distance}(TS, green_shp)",
                              f"{self.distance}(TS, red_shp)", self.threshold_original)

    def get_graphviz_rule(self, columns_names=None, scaler=None, float_precision: Optional[int] = 3):
        """
        Attributes of the Graphviz node of the split: the rule above a plot of the two shapelets, green
        and red, drawn into a temporary PNG file (requires matplotlib).

        Parameters
        ----------
        columns_names : list of str, default=None
            Not used.
        scaler : TransformerMixin, default=None
            Not supported: must be None.
        float_precision : int, default=3
            Not used.

        Returns
        -------
        attributes : dict
            Attributes of the Graphviz node.

        Raises
        ------
        MissingOptionalDependencyError
            If matplotlib is not installed.
        """
        label = self.get_rule(columns_names, scaler, float_precision)["graphviz_rule"]["label"]
        return graphviz_shapelets(label, [self.st.shapelets[0, 0], self.st.shapelets[1, 0]],
                                  colors=list(PROXIMITY_COLORS))


class PartialProximityTreeStump:
    """
    Mixin of the proximity stumps for tabular data (PartialProximityTreeStumpClassifier): the
    PartialProximityTransformer built from the parameters of the stump and the rule.
    """

    def _make_transformer(self):
        """
        A new, unfitted PartialProximityTransformer with the parameters of the stump.

        Returns
        -------
        transformer : PartialProximityTransformer
        """
        return PartialProximityTransformer(n_shapelets=self.n_shapelets,
                                           n_ts_for_selection=self.n_ts_for_selection,
                                           n_features_strategy=self.n_features_strategy,
                                           selection=self.selection,
                                           distance=self.distance,
                                           proximity_on_same_features=self.proximity_on_same_features,
                                           use_combination=self.use_combination,
                                           random_state=self.random_state,
                                           n_jobs=self.n_jobs)

    def get_rule(self, columns_names=None, scaler=None, float_precision: Optional[int] = 3):
        """
        Rule of the split, e.g. ``euclidean(x, SP0(d: petal length=1.3)) <= euclidean(x, SP1(d: petal
        length=5.1))``: the instances closer to the first partial pivot than to the second one go left.

        Parameters
        ----------
        columns_names : list of str, default=None
            Names of the features. If None, ``Feat. <index>``.
        scaler : TransformerMixin, default=None
            Scaler used to preprocess the data: the values of the pivots are shown in the original scale.
        float_precision : int, default=3
            Not used: the values of the pivots are not rounded.

        Returns
        -------
        rule : dict
            See RuleTreeBaseStump.get_rule.
        """
        # imported here: the classification package imports this module
        from RuleTree.stumps.classification.PartialPivotTreeStumpClassifier import describe_partial_pivot

        first, second = (f"{self.distance}(x, SP{i}({describe_partial_pivot(pivot, columns_names, scaler)}))"
                         for i, pivot in enumerate(self.st.shapelets[:2]))
        return proximity_rule(self.feature_original, first, second, self.threshold_original)
