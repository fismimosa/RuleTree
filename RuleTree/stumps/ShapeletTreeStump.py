"""ShapeletTreeStump: what the shapelet stumps for classification and regression have in common."""

from typing import Optional

from RuleTree.base.RuleTreeBaseStump import numerical_rule
from RuleTree.stumps.transformers.ShapeletTransformer import ShapeletTransformer


class ShapeletTreeStump:
    """
    Mixin of the shapelet stumps (ShapeletTreeStumpClassifier and ShapeletTreeStumpRegressor): the
    ShapeletTransformer built from the parameters of the stump, the parameters of the splitter, the
    rule and its drawing.

    The subclass defines the parameters, by deriving also from the decision tree stump of its task
    (``class ShapeletTreeStumpClassifier(ShapeletTreeStump, DecisionTreeStumpClassifier)``).
    """

    def _splitter_params(self):
        """
        Parameters of the splitter: as in the decision tree stump, but without random_state, which is
        the one of the shapelet selection.
        """
        return super()._splitter_params() | {"random_state": None}

    def _make_transformer(self):
        """
        A new, unfitted ShapeletTransformer with the parameters of the stump.

        Returns
        -------
        transformer : ShapeletTransformer
        """
        return ShapeletTransformer(n_shapelets=self.n_shapelets,
                                   n_shapelets_for_selection=self.n_shapelets_for_selection,
                                   n_ts_for_selection_per_class=self.n_ts_for_selection_per_class,
                                   sliding_window=self.sliding_window,
                                   selection=self.selection,
                                   distance=self.distance,
                                   mi_n_neighbors=self.mi_n_neighbors,
                                   random_state=self.random_state,
                                   n_jobs=self.n_jobs,
                                   ml_task=self.task)

    @property
    def st(self):
        """The fitted shapelet transform."""
        return self.transformer_.st_

    def get_rule(self, columns_names=None, scaler=None, float_precision: Optional[int] = 3):
        """
        Rule of the split, e.g. ``euclidean(TS, shp) <= 1.2``.

        Parameters
        ----------
        columns_names : list of str, default=None
            Not used.
        scaler : TransformerMixin, default=None
            Not supported: must be None.
        float_precision : int, default=3
            Number of decimals of the threshold. If None, it is not rounded.

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

        rule = numerical_rule(self.feature_original, f"{self.distance}(TS, shp)", self.threshold_original,
                              float_precision)
        return rule | {"feature_name": f"Shapelet_{self.feature_original}"}

    def get_plot_rule(self, columns_names=None, scaler=None, float_precision: Optional[int] = 3):
        """
        What the node of the split shows in ``RuleTree.plot_tree``: the rule and, below it, the
        shapelet.

        Parameters
        ----------
        columns_names : list of str, default=None
            Not used.
        scaler : TransformerMixin, default=None
            Not supported: must be None.
        float_precision : int, default=3
            Number of decimals of the threshold. If None, it is not rounded.

        Returns
        -------
        rule : dict
            ``label``, ``media_values`` (the shapelet), ``media_kind`` and ``media_names``.
        """
        label = self.get_rule(columns_names, scaler, float_precision)["textual_rule"]
        return {"label": label, "media_values": [self.st.shapelets[self.feature_original, 0]], "media_kind": "series",
                "media_names": ["shp"]}
