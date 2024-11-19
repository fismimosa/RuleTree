from ruletree.base.RuleTreeBaseStump import RuleTreeBaseStump
from ruletree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from ruletree.stumps.splitters.PivotSplit import PivotSplit
from ruletree.utils import MODEL_TYPE_CLF


class PivotTreeStumpClassifier(DecisionTreeStumpClassifier, RuleTreeBaseStump):
    def __init__(self, distance_matrix=None, distance_measure='euclidean', **kwargs):
        super().__init__(**kwargs)
        self.pivot_split = PivotSplit(ml_task=MODEL_TYPE_CLF, **kwargs)
        self.distance_matrix = distance_matrix
        self.distance_measure = distance_measure

    def fit(self, X, y, distance_matrix, distance_measure, idx, sample_weight=None, check_input=True):
        self.feature_analysis(X, y)
        self.num_pre_transformed = self.numerical
        self.cat_pre_transformed = self.categorical
        best_info_gain = -float('inf')

        if len(self.numerical) > 0:
            self.pivot_split.fit(X[:, self.numerical], y, distance_matrix, distance_measure, idx,
                                 sample_weight=sample_weight, check_input=check_input)
            X_transform = self.pivot_split.transform(X[:, self.numerical])
            candidate_names = self.pivot_split.get_candidates_names()
            super().fit(X_transform, y, sample_weight=sample_weight, check_input=check_input)

            self.feature_original = [f'{candidate_names[self.tree_.feature[0]]}_P', -2, -2]
            self.threshold_original = self.tree_.threshold
            self.is_pivotal = True

        return self

    def apply(self, X):
        X_transformed = self.pivot_split.transform(X[:, self.num_pre_transformed], self.distance_measure)
        return super().apply(X_transformed)

    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        raise NotImplementedError()

    def node_to_dict(self, col_names):
        raise NotImplementedError()

    def export_graphviz(self, graph=None, columns_names=None, scaler=None, float_precision=3):
        raise NotImplementedError()