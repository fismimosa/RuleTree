from abc import abstractmethod, ABC

from ruletree.stumps.splitters.ObliqueBivariateSplit import ObliqueBivariateSplit
from ruletree.stumps.splitters.ObliqueHouseHolderSplit import ObliqueHouseHolderSplit
from ruletree.stumps.splitters.PivotSplit import PivotSplit


class ObliquePivotSplit(PivotSplit, ABC):
    def __init__(
            self,
            oblique_split_type='householder',
            **kwargs
    ):
        super().__init__(**kwargs)
        self.oblique_split_type = oblique_split_type

    def get_base_model(self):
        if self.oblique_split_type == 'householder':
            return ObliqueHouseHolderSplit(ml_task=self.ml_task, **self.kwargs)
        if self.oblique_split_type == 'bivariate':
            return ObliqueBivariateSplit(ml_task=self.ml_task, **self.kwargs)

    def compute_discriminative(self, sub_matrix, y, sample_weight=None, check_input=True):
        disc = self.get_base_model()
        disc.fit(sub_matrix, y, sample_weight=sample_weight, check_input=check_input)
        discriminative_id = disc.feats
        return (discriminative_id)
