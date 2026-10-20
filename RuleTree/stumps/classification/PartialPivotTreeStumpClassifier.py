"""PartialPivotTreeStumpClassifier: the split on the distance from a partial pivot, for classification."""

from typing import Optional

import numpy as np

from RuleTree.base.RuleTreeBaseStump import numerical_rule
from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from RuleTree.stumps.transformers.PartialPivotTransformer import PartialPivotTransformer


class PartialPivotTreeStumpClassifier(DecisionTreeStumpClassifier):
    """
    Partial pivot stump for classification of tabular data: a threshold on the distance from a
    *partial pivot*, an instance of the node restricted to a few features (``distance <= t`` goes
    left).

    At each node, ``n_shapelets`` partial pivots are selected among the instances of the node, each on
    a subset of the features (``n_features_strategy``), and the split is the best threshold on the
    distance from one of them. The distance is computed only on the features of the pivot, so the
    rule involves a few features. In the rules, ``SP<i>(...)`` lists the features and values of the
    pivot.

    It is a :class:`~RuleTree.stumps.Stump.Stump` made of a PartialPivotTransformer and a
    ClassificationSplitter.

    Parameters
    ----------
    n_shapelets : int or numpy.inf, default=8
        Number of partial pivots selected at each node (numpy.inf with selection='all').
    n_ts_for_selection : int, float or numpy.inf, default=100
        Number (int) or fraction (float) of the instances of the node used to select the pivots.
    n_features_strategy : int, float, tuple or str, default=2
        Number of features of each partial pivot: an int, a fraction of the features (float), a range
        ``(min, max)``, or chosen from the feature importances of a random forest ('elbow', 'drop',
        'new') or iteratively with depth-1 trees ('iterative').
    selection : {'random', 'cluster', 'all'}, default='random'
        How the pivots are selected among the candidates: at random, the instances closest to the
        k-means centroids, or all of them.
    distance : {'euclidean', 'sqeuclidean', 'cosine', 'cityblock'} or callable, default='euclidean'
        Distance between an instance and a partial pivot, on the features of the pivot. The data are
        used as they are: scale them beforehand if the features have different scales.
    use_combination : bool, default=True
        With the strategies based on feature importances, whether to use all the combinations of the
        important features or only the set of all of them.
    random_state : int, default=42
        Seed of the random choices.
    n_jobs : int, default=1
        Number of threads used to compute the distances (-1: all the CPUs).
    criterion : {'gini', 'entropy'}, default='gini'
        Impurity measure.
    min_samples_leaf : int or float, default=1
        Minimum number of instances in each child (a fraction of the instances of the node if
        float). Applies to the numerical features.
    min_weight_fraction_leaf : float, default=0.0
        Minimum fraction of the sample weights of the node in each child. Applies to the numerical
        features.
    max_features : int, float, {'sqrt', 'log2'} or None, default=None
        Number of numerical features considered for the split, drawn at random; None for all.
    min_impurity_decrease : float, default=0.0
        A numerical split is kept only if it decreases the weighted impurity by at least this value.
    class_weight : dict, 'balanced' or None, default=None
        Weights of the classes in the impurity; 'balanced' weighs them inversely to their frequency.

    Attributes
    ----------
    transformer_ : PartialPivotTransformer
        The fitted transformer.
    splitter_ : ClassificationSplitter
        The fitted splitter (feature, threshold, gain and impurity of the split).
    data_type_ : str
        Data type the stump was fitted on.
    feature_original : int
        Index of the partial pivot of the split.
    threshold_original : float
        Threshold of the split.

    See Also
    --------
    PivotTreeStumpClassifier : pivots on all the features.
    RuleTree.utils.shapelet_transform.TabularShapelets.TabularShapelets : the selection of the pivots.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from RuleTree.stumps.classification import PartialPivotTreeStumpClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> stump = PartialPivotTreeStumpClassifier(n_shapelets=10, n_features_strategy=2, random_state=0).fit(X, y)
    >>> stump.get_rule(float_precision=2)["textual_rule"]
    'SP0(d: Feat. 2=1.3, Feat. 3=0.2) <= 1.28'
    """

    def __init__(self,
                 n_shapelets=8,
                 n_ts_for_selection=100,  # int, inf
                 n_features_strategy=2,
                 selection='random',  # random, cluster, all
                 distance='euclidean',
                 use_combination=True,
                 random_state=42, n_jobs=1,
                 criterion='gini', min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                 min_impurity_decrease=0.0, class_weight=None):
        """Initialize the PartialPivotTreeStumpClassifier; see the class documentation for the parameters."""
        if selection not in ["random", "cluster", "all"]:
            raise ValueError("'selection' must be 'random', 'all' or 'cluster'")

        self.n_shapelets = n_shapelets
        self.n_ts_for_selection = n_ts_for_selection
        self.n_features_strategy = n_features_strategy
        self.selection = selection
        self.distance = distance
        self.use_combination = use_combination
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight

    def _splitter_params(self):
        """
        Parameters of the splitter: as in the decision tree stump, but without random_state, which is
        the one of the selection of the partial pivots.

        Returns
        -------
        params : dict
        """
        return super()._splitter_params() | {"random_state": None}

    def _make_transformer(self):
        """
        A new, unfitted PartialPivotTransformer with the parameters of the stump.

        Returns
        -------
        transformer : PartialPivotTransformer
        """
        return PartialPivotTransformer(n_shapelets=self.n_shapelets,
                                       n_ts_for_selection=self.n_ts_for_selection,
                                       n_features_strategy=self.n_features_strategy,
                                       selection=self.selection,
                                       distance=self.distance,
                                       use_combination=self.use_combination,
                                       random_state=self.random_state,
                                       n_jobs=self.n_jobs)

    @property
    def st(self):
        """The fitted transform of the partial pivots."""
        return self.transformer_.st_

    def get_rule(self, columns_names=None, scaler=None, float_precision: Optional[int] = 3):
        """
        Rule of the split, e.g. ``SP0(d: sepal length=5.1, petal width=0.2) <= 0.8``: the distance from
        the partial pivot, described by the values of its features.

        Parameters
        ----------
        columns_names : list of str, default=None
            Names of the features. If None, ``Feat. <index>``.
        scaler : TransformerMixin, default=None
            Scaler used to preprocess the data: the values of the pivot are shown in the original scale.
        float_precision : int, default=3
            Number of decimals of the threshold. If None, it is not rounded.

        Returns
        -------
        rule : dict
            See RuleTreeBaseStump.get_rule.
        """
        description = describe_partial_pivot(self.st.shapelets[self.feature_original], columns_names, scaler)
        return numerical_rule(self.feature_original, f"SP{self.feature_original}({description})",
                              self.threshold_original, float_precision)


def describe_partial_pivot(pivot, columns_names=None, scaler=None) -> str:
    """
    Description of a partial pivot by the values of its features, e.g.
    ``d: petal length=1.3, petal width=0.2``.

    Parameters
    ----------
    pivot : numpy.ndarray of shape (n_features,)
        The partial pivot: nan for the features it does not use.
    columns_names : list of str, default=None
        Names of the features. If None, ``Feat. <index>``.
    scaler : TransformerMixin, default=None
        Scaler used to preprocess the data: the values are shown in the original scale.

    Returns
    -------
    description : str
        The features used by the pivot and their values; if it uses more than half of the
        features, the ones it does not use.
    """
    if columns_names is None:
        columns_names = [f"Feat. {i}" for i in range(pivot.shape[0])]

    pivot = np.copy(pivot)
    if scaler is not None:
        pivot = scaler.inverse_transform(pivot.reshape(1, -1)).flatten()
    frac_not_null = 1 - np.isnan(pivot).sum() / pivot.shape[0]
    if frac_not_null <= .50:
        feature_list = [f'{name}={val}' for name, val in zip(columns_names, pivot) if not np.isnan(val)]
        return 'd: ' + (", ".join(feature_list))
    feature_list = [f'{name}={val}' for name, val in zip(columns_names, pivot) if np.isnan(val)]
    return f"d all excpet: {', '.join(feature_list)}"
