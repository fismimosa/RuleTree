from ruletree.base.RuleTreeBaseStump import RuleTreeBaseStump
from ruletree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from ruletree.stumps.splitters.MultiplePivotSplit import MultiplePivotSplit
from ruletree.utils import MODEL_TYPE_CLF


class MultiplePivotTreeStumpClassifier(DecisionTreeStumpClassifier, RuleTreeBaseStump):
    def __init__(self, distance_matrix=None, distance_measure='euclidean', **kwargs):
        super().__init__(**kwargs)
        self.multi_pivot_split = MultiplePivotSplit(ml_task=MODEL_TYPE_CLF, **kwargs)
        self.distance_matrix = distance_matrix
        self.distance_measure = distance_measure

    def fit(self, X, y, distance_matrix, distance_measure, idx, sample_weight=None, check_input=True):
        self.feature_analysis(X, y)
        self.num_pre_transformed = self.numerical
        self.cat_pre_transformed = self.categorical

        if len(self.numerical) > 0:
            self.multi_pivot_split.fit(X[:, self.numerical], y, distance_matrix, distance_measure, idx,
                                       sample_weight=sample_weight, check_input=check_input)
            X_transform = self.multi_pivot_split.transform(X[:, self.numerical])
            candidate_names = self.multi_pivot_split.get_candidates_names()
            super().fit(X_transform, y, sample_weight=sample_weight, check_input=check_input)

            self.feature_original = [(candidate_names[0], candidate_names[1]), -2, -2]
            self.threshold_original = self.tree_.threshold
            self.is_pivotal = True

        return self

    def apply(self, X):
        X_transformed = self.multi_pivot_split.transform(X[:, self.num_pre_transformed], self.distance_measure)
        return super().apply(X_transformed)

    def get_rule(self, columns_names=None, scaler=None, float_precision=3):
        raise NotImplementedError()

    def node_to_dict(self, col_names):
        raise NotImplementedError()

    def export_graphviz(self, graph=None, columns_names=None, scaler=None, float_precision=3):
        raise NotImplementedError()
