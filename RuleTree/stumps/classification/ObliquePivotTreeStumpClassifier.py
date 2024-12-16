from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump
from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from RuleTree.stumps.splitters.ObliqueBivariateSplit import ObliqueBivariateSplit
from RuleTree.stumps.splitters.ObliqueHouseHolderSplit import ObliqueHouseHolderSplit
from RuleTree.stumps.splitters.ObliquePivotSplit import ObliquePivotSplit
from RuleTree.utils import MODEL_TYPE_CLF


class ObliquePivotTreeStumpClassifier(DecisionTreeStumpClassifier, RuleTreeBaseStump):
    def __init__(self,
                 distance_matrix=None,
                 distance_measure='euclidean',
                 oblique_split_type='householder',
                 pca=None,
                 max_oblique_features=2,
                 tau=1e-4,
                 n_orientations=10,
                 **kwargs):
        super().__init__(**kwargs)
        self.distance_matrix = distance_matrix
        self.distance_measure = distance_measure

        self.pca = pca
        self.max_oblique_features = max_oblique_features
        self.tau = tau
        self.n_orientations = n_orientations

        self.obl_pivot_split = ObliquePivotSplit(ml_task=MODEL_TYPE_CLF, oblique_split_type=oblique_split_type, **kwargs)

        if oblique_split_type == 'householder':
            self.oblique_split = ObliqueHouseHolderSplit(pca=self.pca,
                                                         max_oblique_features=self.max_oblique_features,
                                                         tau=self.tau,
                                                         **kwargs)

        if oblique_split_type == 'bivariate':
            self.oblique_split = ObliqueBivariateSplit(ml_task=MODEL_TYPE_CLF, n_orientations=self.n_orientations, **kwargs)

    def fit(self, X, y, distance_matrix, distance_measure, idx, sample_weight=None, check_input=True):
        self.feature_analysis(X, y)
        self.num_pre_transformed = self.numerical
        self.cat_pre_transformed = self.categorical

        if len(self.numerical) > 0:
            self.obl_pivot_split.fit(X[:, self.numerical], y, distance_matrix, distance_measure, idx,
                                     sample_weight=sample_weight, check_input=check_input)
            X_transform = self.obl_pivot_split.transform(X[:, self.numerical])
            candidate_names = self.obl_pivot_split.get_candidates_names()

            self.oblique_split.fit(X_transform, y, sample_weight=sample_weight, check_input=check_input)
            X_transform_oblique = self.oblique_split.transform(X_transform)
            super().fit(X_transform_oblique, y, sample_weight=sample_weight, check_input=check_input)

            feats = [f'{p}_P' for p in candidate_names[self.oblique_split.feats]]
            self.feature_original = [feats, -2, -2]
            self.coefficients = self.oblique_split.coeff
            self.threshold_original = self.tree_.threshold
            self.is_oblique = True
            self.is_pivotal = True

        return self

    def apply(self, X):
        X_transformed = self.obl_pivot_split.transform(X[:, self.num_pre_transformed], self.distance_measure)
        X_transformed_oblique = self.oblique_split.transform(X_transformed)
        return super().apply(X_transformed_oblique)

    def get_params(self, deep=True):
        return {
            **self.kwargs,
            'max_oblique_features': self.max_oblique_features,
            'pca': self.pca,
            'tau': self.tau,
            'n_orientations': self.n_orientations
        }

    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        raise NotImplementedError()

    def node_to_dict(self, col_names):
        raise NotImplementedError()

    def export_graphviz(self, graph=None, columns_names=None, scaler=None, float_precision=3):
        raise NotImplementedError()
