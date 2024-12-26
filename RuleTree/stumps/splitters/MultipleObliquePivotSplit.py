import itertools

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.tree import DecisionTreeClassifier
from inspect import signature

from RuleTree.stumps.splitters.ObliquePivotSplit import ObliquePivotSplit
from RuleTree.utils import MODEL_TYPE_CLU, MODEL_TYPE_REG, MODEL_TYPE_CLF
from RuleTree.utils.data_utils import get_info_gain


class MultipleObliquePivotSplit(ObliquePivotSplit):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.best_tup = None
        self.best_tup_name = None
        self.best_gain = -float('inf')

    def find_best_tuple(self, X, y, distance_measure='euclidean', sample_weight=None, check_input=True):
        two_tuples = list(itertools.combinations(range(0, len(self.X_candidates)), 2))
    
        for tup in two_tuples:
        
            if len(set(self.y_candidates[np.array(tup)])) == 1:
                continue
            disc = self.get_base_model_for_tuple_finding()
           
                    
            p1, p2 = self.X_candidates[np.array(tup)]
            name_p1, name_p2 = self.candidates_names[np.array(tup)]

            dist_to_p0 = pairwise_distances(X, p1.reshape(1, -1), metric=distance_measure).flatten()
            dist_to_p1 = pairwise_distances(X, p2.reshape(1, -1), metric=distance_measure).flatten()

            dist_binary = np.where(dist_to_p0 < dist_to_p1, 0, 1).reshape(-1, 1)
            disc.fit(dist_binary, y)
            gain_disc = get_info_gain(disc)

            if gain_disc > self.best_gain:
                self.best_gain = gain_disc
                self.best_tup = self.X_candidates[np.array(tup)]
                self.best_tup_name = self.candidates_names[np.array(tup)]

    def fit(self, X, y, distance_matrix, distance_measure, idx,
            sample_weight=None, check_input=True):

        super().fit(X, y, distance_matrix, distance_measure, idx, sample_weight=sample_weight, check_input=check_input)
        self.find_best_tuple(X, y, distance_measure=distance_measure, sample_weight=sample_weight,
                             check_input=check_input)

    def transform(self, X, distance_measure='euclidean'):
        dist_to_p0 = pairwise_distances(X, self.best_tup[0].reshape(1, -1), metric=distance_measure).flatten()
        dist_to_p1 = pairwise_distances(X, self.best_tup[1].reshape(1, -1), metric=distance_measure).flatten()
        dist_binary = np.where(dist_to_p0 < dist_to_p1, 0, 1).reshape(-1, 1)
        return dist_binary

    def get_best_tup_names(self):
        return self.best_tup_name

    def get_base_model_for_tuple_finding(self):
        if self.ml_task == MODEL_TYPE_CLF:
            valid_params = set(signature(DecisionTreeClassifier).parameters.keys())
            filtered_kwargs = {key: value for key, value in self.kwargs.items() if key in valid_params}
            return DecisionTreeClassifier(**filtered_kwargs)
        elif self.ml_task == MODEL_TYPE_REG:
            raise NotImplementedError("Regression is not implemented yet.")
        elif self.ml_task == MODEL_TYPE_CLU:
            raise NotImplementedError("Clustering is not implemented yet.")
