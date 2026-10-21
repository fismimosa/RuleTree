"""Selection of the candidate pivots of a node, used by PivotTransformer."""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from RuleTree.exceptions import NoSplitFoundWarning
from RuleTree.utils.define import MODEL_TYPE_REG


class PivotSelection:
    """
    Selection of the candidate pivots among the instances of a node, from their distance matrix.

    For classification, the candidates are, for each class, its medoid (descriptive pivot) and the
    instance whose distance best separates the classes, found by a depth-1 decision tree on the
    distances (discriminative pivot). For regression, the descriptive pivot is the instance whose
    distance best splits the target and best contrasts close and distant instances, and the
    discriminative pivot is found by a depth-1 regression tree.

    With an oblique ``discriminative``, the discriminative pivots are instead the instances combined by
    an oblique direction on the distances (Householder or bivariate, see the oblique transformers):
    up to ``max_oblique_features`` (Householder) or two (bivariate) pivots for each class.

    Parameters
    ----------
    ml_task : {MODEL_TYPE_CLF, MODEL_TYPE_REG}
        Task of the stump.
    k : int, default=None
        Regression only: number of nearest and farthest instances used to measure the contrast of a
        pivot. None for the square root of the number of instances (between 5 and n - 1).
    reg_contrast_lambda : float, default=0.2
        Regression only: weight of the contrast with respect to the gain.
    reg_max_thresholds : int, default=None
        Regression only: maximum number of thresholds tested on the distances from each pivot. None
        (or not finite) for ``max(256, sqrt(n))``.
    discriminative : {'tree', 'householder', 'bivariate'}, default='tree'
        How the discriminative pivots are found: the root split of a decision tree, or the instances
        combined by an oblique direction.
    max_oblique_features : int, default=2
        Householder only: maximum number of pivots combined by the reflection.
    tau : float, default=1e-4
        Householder only: threshold of the reflection (see HouseholderTransformer).
    n_orientations : int, default=10
        Bivariate only: number of angles (see BivariateTransformer).
    **tree_params : dict
        Parameters of the decision trees that find the discriminative pivots.

    Attributes
    ----------
    candidates_ : numpy.ndarray of shape (n_candidates, n_features)
        The candidate pivots: the descriptive ones, then the discriminative ones.
    candidate_names_, descriptive_names_, discriminative_names_ : numpy.ndarray of int
        Indices of the candidate, descriptive and discriminative pivots (as given by ``idx``).
    candidate_labels_ : numpy.ndarray
        Target of the candidate pivots.
    """

    def __init__(self, ml_task, k=None, reg_contrast_lambda=.2, reg_max_thresholds=None, discriminative='tree',
                 max_oblique_features=2, tau=1e-4, n_orientations=10, **tree_params):
        """Initialize the selection; see the class documentation for the parameters."""
        self.ml_task = ml_task
        self.discriminative = discriminative
        self.max_oblique_features = max_oblique_features
        self.tau = tau
        self.n_orientations = n_orientations
        self.k = k
        self.reg_contrast_lambda = reg_contrast_lambda
        try:
            self.reg_max_thresholds = reg_max_thresholds if np.isfinite(reg_max_thresholds) else None
        except TypeError:
            self.reg_max_thresholds = None
        self.tree_params = tree_params

    def _base_model(self):
        """
        Decision tree of the task, with tree_params, whose root split finds the discriminative pivot. Only
        the root is needed, so the tree has depth 1: scikit-learn chooses the split of the root before
        the rest of the tree, so the root is the same as with a deeper tree.
        """
        tree_params = self.tree_params | {"max_depth": 1}
        if self.ml_task == MODEL_TYPE_REG:
            return DecisionTreeRegressor(**tree_params)
        return DecisionTreeClassifier(**tree_params)

    def compute_descriptive(self, sub_matrix):
        """
        Computes the descriptive (most central) instance for a class.

        This method identifies the medoid of a class, which is the instance
        with the minimum sum of distances to all other instances in the same class.

        Parameters
        ----------
        sub_matrix : array-like
            Distance matrix for instances of a particular class.

        Returns
        -------
        int
            Index of the medoid (instance with minimum sum of distances to other instances).
        """
        row_sums = sub_matrix.sum(axis=1)
        medoid_index = np.argmin(row_sums)
        return medoid_index

    def _resolve_knn_density_k(self, n_samples):
        """
        Number of neighbours used to measure the contrast of a pivot (regression).

        Parameters
        ----------
        n_samples : int
            Number of instances in the node.

        Returns
        -------
        k : int
            ``k``, or the square root of n_samples if k is None, between 5 and n_samples - 1.
        """
        if self.k is None:
            k = int(np.sqrt(n_samples))
        else:
            k = int(self.k)

        k = min(max(5, k), n_samples)
        if n_samples <= 1:
            return 1
        return min(k, n_samples - 1)

    @staticmethod
    def _root_split_gain(tree_):
        """
        Decrease of impurity of the root split of a fitted scikit-learn tree.

        Parameters
        ----------
        tree_ : sklearn.tree._tree.Tree
            Fitted tree.

        Returns
        -------
        gain : float
            0 if the tree has no split.
        """
        if tree_.feature[0] == -2:
            return 0.0

        w = tree_.weighted_n_node_samples
        imp = tree_.impurity
        if w[0] <= 0:
            return 0.0

        return float(imp[0] - (w[1] / w[0]) * imp[1] - (w[2] / w[0]) * imp[2])

    def compute_descriptive_regression(self, sub_matrix, y, sample_weight=None):
        """
        Descriptive pivot for regression: the instance whose distance best separates the target (gain of a
        depth-1 regression tree on the distances) and best contrasts near and far instances.

        Parameters
        ----------
        sub_matrix : numpy.ndarray of shape (n_samples, n_samples)
            Distances between the instances of the node.
        y : numpy.ndarray
            Target of the instances of the node.
        sample_weight : numpy.ndarray, default=None
            Sample weights.

        Returns
        -------
        pivot : int
            Index (in the node) of the descriptive pivot.
        """
        n_samples = len(y)
        if n_samples <= 1:
            return 0

        k = self._resolve_knn_density_k(n_samples)

        gains = np.zeros(n_samples, dtype=float)
        contrasts = np.zeros(n_samples, dtype=float)

        for pivot_idx in range(n_samples):
            distances = sub_matrix[:, pivot_idx]
            sorted_distances = np.sort(distances)

            near_distances = sorted_distances[1:k + 1]
            far_distances = sorted_distances[-k:]
            if len(near_distances) == 0:
                near_distances = sorted_distances[:1]
            if len(far_distances) == 0:
                far_distances = sorted_distances[-1:]
            contrasts[pivot_idx] = float(np.mean(far_distances) - np.mean(near_distances))

            distances_for_gain = distances
            max_thresholds = int(self.reg_max_thresholds) if self.reg_max_thresholds is not None else (
                max(256, int(np.sqrt(n_samples))))
            unique_distances = np.unique(distances)
            if max_thresholds > 0 and len(unique_distances) - 1 > max_thresholds:
                quantiles = np.linspace(0.0, 1.0, max_thresholds + 2)[1:-1]
                bins = np.unique(np.quantile(distances, quantiles))
                if len(bins) > 0:
                    distances_for_gain = np.digitize(distances, bins=bins, right=True).astype(float)

            reg = DecisionTreeRegressor(
                criterion=self.tree_params.get("criterion", "squared_error"),
                max_depth=1,
                min_samples_split=self.tree_params.get("min_samples_split", 2),
                min_samples_leaf=self.tree_params.get("min_samples_leaf", 1),
                random_state=self.tree_params.get("random_state", None),
            )
            reg.fit(distances_for_gain.reshape(-1, 1), y, sample_weight=sample_weight)
            gains[pivot_idx] = self._root_split_gain(reg.tree_)

        gains_min, gains_max = np.min(gains), np.max(gains)
        contrast_min, contrast_max = np.min(contrasts), np.max(contrasts)

        gains_norm = (gains - gains_min) / (gains_max - gains_min + 1e-12)
        contrasts_norm = (contrasts - contrast_min) / (contrast_max - contrast_min + 1e-12)

        scores = gains_norm + float(self.reg_contrast_lambda) * contrasts_norm
        return int(np.argmax(scores))

    def compute_discriminative(self, sub_matrix, y, sample_weight=None, check_input=True):
        """
        Computes the discriminative instance for a class.

        This method identifies the instance that best separates different classes
        by training a decision tree and extracting the feature used at the root node.

        Parameters
        ----------
        sub_matrix : array-like
            Distance matrix for instances of a particular class.
        y : array-like
            Target values.
        sample_weight : array-like, optional
            Sample weights for weighted learning.
        check_input : bool, default=True
            Whether to validate input.

        Returns
        -------
        list of int
            Indices (columns of sub_matrix) of the discriminative instances: the root feature of the
            tree, or the instances combined by the oblique direction. Empty if there is no split.
        """
        if self.discriminative == "tree":
            disc = self._base_model()
            disc.fit(sub_matrix, y, sample_weight=sample_weight, check_input=check_input)
            discriminative_id = disc.tree_.feature[0]
            return [] if discriminative_id == -2 else [int(discriminative_id)]  # -2: the tree did not split

        oblique = self._oblique_model()
        try:
            # distances of the node from the candidates: no cache of the tree
            oblique.fit(sub_matrix, y, sample_weight=sample_weight, context=None, node_id=None)
        except NoSplitFoundWarning:
            return []
        return oblique.oblique_features()

    def _oblique_model(self):
        """Oblique transformer of ``discriminative``, whose directions find the discriminative pivots."""
        # imported here: the transformers package imports this module
        from RuleTree.stumps.transformers.BivariateTransformer import BivariateTransformer
        from RuleTree.stumps.transformers.HouseholderTransformer import HouseholderTransformer

        if self.discriminative == "householder":
            return HouseholderTransformer(max_oblique_features=self.max_oblique_features, tau=self.tau,
                                          random_state=self.tree_params.get("random_state"))
        if self.discriminative == "bivariate":
            return BivariateTransformer(self.ml_task, n_orientations=self.n_orientations, tree_params=self.tree_params)
        raise ValueError(f"Unknown discriminative {self.discriminative!r}: 'tree', 'householder' or 'bivariate'")

    def fit(self, X, y, distance_matrix, idx, sample_weight=None, check_input=True):
        """
        Select the candidate pivots of a node.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Instances of the node.
        y : numpy.ndarray of shape (n_samples,)
            Target of the instances of the node.
        distance_matrix : numpy.ndarray of shape (n_samples, n_samples)
            Distances between the instances of the node.
        idx : numpy.ndarray of shape (n_samples,)
            Names of the instances (their indices in the training set).
        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Sample weights.
        check_input : bool, default=True
            Whether the decision trees check their input.

        Returns
        -------
        self : PivotSelection
        """
        descriptives, discriminatives = [], []
        if self.ml_task == MODEL_TYPE_REG:
            descriptives = [self.compute_descriptive_regression(distance_matrix, y, sample_weight=sample_weight)]
            discriminatives = self.compute_discriminative(distance_matrix, y, sample_weight=sample_weight,
                                                          check_input=check_input)
        else:
            for label in np.unique(y):  # sorted: set(y) depends on the hash seed with string labels
                idx_label = np.where(y == label)[0]
                sub_matrix_label = distance_matrix[:, idx_label]
                disc_ids = self.compute_discriminative(sub_matrix_label, y, sample_weight=sample_weight,
                                                       check_input=check_input)
                discriminatives += idx_label[disc_ids].tolist()
                descriptives.append(idx_label[self.compute_descriptive(sub_matrix_label[idx_label])])

        candidates = descriptives + discriminatives
        self.candidates_ = X[candidates]
        self.descriptive_names_ = idx[descriptives]
        self.discriminative_names_ = idx[discriminatives]
        self.candidate_names_ = idx[candidates]
        self.candidate_labels_ = np.asarray(y)[candidates]
        return self
