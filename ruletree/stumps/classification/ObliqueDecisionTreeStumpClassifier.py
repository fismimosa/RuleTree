from ruletree.base.RuleTreeBaseStump import RuleTreeBaseStump
from ruletree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from ruletree.stumps.splitters.ObliqueBivariateSplit import ObliqueBivariateSplit
from ruletree.stumps.splitters.ObliqueHouseHolderSplit import ObliqueHouseHolderSplit
from ruletree.utils import MODEL_TYPE_CLF


class ObliqueDecisionTreeStumpClassifier(DecisionTreeStumpClassifier, RuleTreeBaseStump):
    def __init__(self,
                 oblique_split_type='householder',
                 pca=None,
                 max_oblique_features=2,
                 tau=1e-4,
                 n_orientations=10,
                 **kwargs):
        super().__init__(**kwargs)
        self.pca = pca
        self.max_oblique_features = max_oblique_features
        self.tau = tau
        self.n_orientations = n_orientations

        if oblique_split_type == 'householder':
            self.oblique_split = ObliqueHouseHolderSplit(pca=self.pca,
                                                         max_oblique_features=self.max_oblique_features,
                                                         tau=self.tau,
                                                         **kwargs)

        if oblique_split_type == 'bivariate':
            self.oblique_split = ObliqueBivariateSplit(ml_task=MODEL_TYPE_CLF, n_orientations=self.n_orientations, **kwargs)

    def fit(self, X, y, sample_weight=None, check_input=True):
        self.feature_analysis(X, y)
        self.num_pre_transformed = self.numerical
        self.cat_pre_transformed = self.categorical
        best_info_gain = -float('inf')

        if len(self.numerical) > 0:
            self.oblique_split.fit(X[:, self.numerical], y, sample_weight=sample_weight, check_input=check_input)
            X_transform = self.oblique_split.transform(X[:, self.numerical])
            super().fit(X_transform, y, sample_weight=sample_weight, check_input=check_input)

            self.feature_original = [self.oblique_split.feats, -2, -2]
            self.coefficients = self.oblique_split.coeff
            self.threshold_original = self.tree_.threshold
            self.is_oblique = True

        return self

    def apply(self, X):
        X_transform = self.oblique_split.transform(X[:, self.num_pre_transformed])
        return super().apply(X_transform)

    def get_params(self, deep=True):
        return {
            **self.kwargs,
            'max_oblique_features': self.max_oblique_features,
            'pca': self.pca,
            'tau': self.tau,
            'n_orientations': self.n_orientations
        }
