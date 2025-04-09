from abc import abstractmethod, ABC

from RuleTree.stumps.splitters.ObliqueBivariateSplit import ObliqueBivariateSplit
from RuleTree.stumps.splitters.ObliqueHouseHolderSplit import ObliqueHouseHolderSplit
from RuleTree.stumps.splitters.PivotSplit import PivotSplit


class ObliquePivotSplit(PivotSplit, ABC):
    """
    ObliquePivotSplit extends PivotSplit by using oblique splits for finding discriminative instances.
    
    This class allows for creating oblique decision boundaries which can better separate classes
    by using either Householder transformations or bivariate splits. Oblique splits can find
    more complex decision boundaries than axis-aligned splits, potentially leading to better
    separation of classes.
    
    Parameters
    ----------
    oblique_split_type : str, default='householder'
        Type of oblique split to use. Options are 'householder' or 'bivariate'.
        - 'householder' uses Householder transformations for finding splits
        - 'bivariate' uses pairs of features for finding splits
    ml_task : str
        The machine learning task type (classification, regression, or clustering).
    **kwargs : dict
        Additional parameters to pass to the base model.
    
    Attributes
    ----------
    oblique_split_type : str
        The type of oblique split to use.
    """
    def __init__(
            self,
            oblique_split_type='householder',
            **kwargs
    ):
        super().__init__(**kwargs)
        self.oblique_split_type = oblique_split_type

    def get_base_model(self):
        """
        Returns the appropriate oblique split model based on the specified type.
        
        Creates and returns either an ObliqueHouseHolderSplit or ObliqueBivariateSplit
        model depending on the configuration.
        
        Returns
        -------
        model : estimator
            Either ObliqueHouseHolderSplit or ObliqueBivariateSplit model.
        """
        if self.oblique_split_type == 'householder':
            return ObliqueHouseHolderSplit(ml_task=self.ml_task, **self.kwargs)
        if self.oblique_split_type == 'bivariate':
            return ObliqueBivariateSplit(ml_task=self.ml_task, **self.kwargs)

    def compute_discriminative(self, sub_matrix, y, sample_weight=None, check_input=True):
        """
        Computes the discriminative features using the selected oblique split method.
        
        This method overrides the parent class method to use oblique splits instead
        of axis-aligned splits for finding discriminative instances.
        
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
        array-like
            Indices of the most discriminative features.
        """
        disc = self.get_base_model()
        disc.fit(sub_matrix, y, sample_weight=sample_weight, check_input=check_input)
        discriminative_id = disc.feats
        return (discriminative_id)
