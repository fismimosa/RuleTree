from abc import abstractmethod, ABC

import numpy as np
from itertools import chain
from sklearn.base import TransformerMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.tree import DecisionTreeClassifier

from ruletree.base.RuleTreeBaseSplit import RuleTreeBaseSplit
from ruletree.base.RuleTreeBaseStump import RuleTreeBaseStump
from ruletree.utils.define import MODEL_TYPE_CLF, MODEL_TYPE_REG, MODEL_TYPE_CLU
import itertools
from ruletree.utils.data_utils import get_info_gain


class PivotSplit(TransformerMixin, RuleTreeBaseSplit, ABC):
    def __init__(
            self,

            ml_task,
            **kwargs
    ):
        super(RuleTreeBaseSplit, RuleTreeBaseSplit).__init__(ml_task)
        self.kwargs = kwargs
        self.X_candidates = None
        self.is_categorical = False
        self.ml_task = ml_task

        self.discriminative_names = None
        self.descriptive_names = None
        self.candidates_names = None
        self.is_pivotal = False

    #@abstractmethod
    def get_base_model(self):
        if self.ml_task == MODEL_TYPE_CLF:
            return DecisionTreeClassifier(**self.kwargs)
        elif self.ml_task == MODEL_TYPE_REG:
            return NotImplementedError()
        elif self.ml_task == MODEL_TYPE_CLU:
            raise NotImplementedError()

    def compute_descriptive(self, sub_matrix):
        row_sums = sub_matrix.sum(axis=1)
        medoid_index = np.argmin(row_sums)
        return medoid_index

    def compute_discriminative(self, sub_matrix, y, sample_weight=None, check_input=True):
        disc = self.get_base_model()
        disc.fit(sub_matrix, y, sample_weight=sample_weight, check_input=check_input)
        discriminative_id = disc.tree_.feature[0]
        return discriminative_id

    def fit(self, X, y, distance_matrix, distance_measure, idx,
            sample_weight=None, check_input=True):

        sub_matrix = distance_matrix
        local_idx = np.arange(len(y))

        local_descriptives = []
        local_discriminatives = []
        local_candidates = []

        for label in set(y):
            idx_label = np.where(y == label)[0]
            local_idx_label = local_idx[idx_label]
            sub_matrix_label = sub_matrix[:, idx_label]

            disc_id = self.compute_discriminative(sub_matrix_label, y, sample_weight=sample_weight,
                                                  check_input=check_input)
            disc_idx = local_idx_label[disc_id]

            desc_id = self.compute_descriptive(sub_matrix_label[idx_label])
            desc_idx = local_idx_label[desc_id]

            if isinstance(disc_idx, (list, np.ndarray)):
                local_discriminatives += disc_idx.flatten().tolist() if isinstance(disc_idx, np.ndarray) else list(
                    disc_idx)
            else:
                local_discriminatives += [disc_idx]

            local_descriptives += [desc_idx]

        local_candidates = local_descriptives + local_discriminatives

        self.X_candidates = X[local_candidates]

        self.discriminative_names = idx[local_discriminatives]
        self.descriptive_names = idx[local_descriptives]
        self.candidates_names = idx[local_candidates]

    def transform(self, X, distance_measure):
        return pairwise_distances(X, self.X_candidates, metric=distance_measure)

    def get_candidates_names(self):
        return self.candidates_names

    def get_descriptive_names(self):
        return self.descriptive_names

    def get_discriminative_names(self):
        return self.discriminative_names
