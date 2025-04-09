from abc import abstractmethod, ABC

import numpy as np
from itertools import chain
from sklearn.base import TransformerMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.tree import DecisionTreeClassifier

from RuleTree.base.RuleTreeBaseSplit import RuleTreeBaseSplit
from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump
from RuleTree.utils.define import MODEL_TYPE_CLF, MODEL_TYPE_REG, MODEL_TYPE_CLU
import itertools
from RuleTree.utils.data_utils import get_info_gain


class PivotSplit(TransformerMixin, RuleTreeBaseSplit, ABC):
    """
    PivotSplit is a split strategy that selects pivot instances to represent classes.
    
    This class identifies both descriptive (most central) and discriminative (most separating)
    instances for each class. These pivot instances are then used as candidates for determining
    the split. Descriptive instances represent the "typical" members of a class, while
    discriminative instances help to distinguish between different classes.
    
    Parameters
    ----------
    ml_task : str
        The machine learning task type (classification, regression, or clustering).
    **kwargs : dict
        Additional parameters to pass to the base model (e.g., DecisionTreeClassifier).

    Attributes
    ----------
    X_candidates : array-like
        Selected candidate instances for splitting.
    is_categorical : bool
        Flag indicating if the features are categorical.
    ml_task : str
        Type of machine learning task.
    discriminative_names : array-like
        Names/indices of discriminative instances.
    descriptive_names : array-like
        Names/indices of descriptive instances.
    candidates_names : array-like
        Names/indices of all candidate instances.
    is_pivotal : bool
        Flag indicating if the split is based on pivot instances.
    """
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
        """
        Returns the appropriate base model based on the machine learning task.
        
        The model is used for finding discriminative split points between classes.

        Returns
        -------
        model : estimator
            The machine learning model to use for finding splits.
            Returns DecisionTreeClassifier for classification tasks.

        Raises
        ------
        NotImplementedError
            If the ml_task is regression or clustering, which are not yet implemented.
        """
        if self.ml_task == MODEL_TYPE_CLF:
            return DecisionTreeClassifier(**self.kwargs)
        elif self.ml_task == MODEL_TYPE_REG:
            return NotImplementedError()
        elif self.ml_task == MODEL_TYPE_CLU:
            raise NotImplementedError()

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
        int
            Index of the most discriminative instance.
        """
        disc = self.get_base_model()
        disc.fit(sub_matrix, y, sample_weight=sample_weight, check_input=check_input)
        discriminative_id = disc.tree_.feature[0]
        return discriminative_id

    def fit(self, X, y, distance_matrix, distance_measure, idx,
            sample_weight=None, check_input=True):
        """
        Fits the PivotSplit by finding descriptive and discriminative instances.
        
        For each class, this method identifies both descriptive instances (those that
        best represent the class) and discriminative instances (those that best separate
        the classes). These instances are stored as candidates for determining splits.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.
        y : array-like of shape (n_samples,)
            Target values.
        distance_matrix : array-like of shape (n_samples, n_samples)
            Pre-computed distance matrix between instances.
        distance_measure : str or callable
            Distance measure to use (e.g., 'euclidean', 'cosine').
        idx : array-like of shape (n_samples,)
            Indices of the instances.
        sample_weight : array-like, optional
            Sample weights.
        check_input : bool, default=True
            Whether to validate input.
            
        Returns
        -------
        self
            The fitted splitter.
        """
        sub_matrix = distance_matrix
        local_idx = np.arange(len(y))

        local_descriptives = []
        local_discriminatives = []
        local_candidates = []

        for label in set(y):
            idx_label = np.where(y == label)[0]
            local_idx_label = local_idx[idx_label]
            sub_matrix_label = sub_matrix[:, idx_label]

            disc_id = self.compute_discriminative(sub_matrix_label, y, 
                                                  sample_weight=sample_weight,
                                                  check_input=check_input)

            desc_id = self.compute_descriptive(sub_matrix_label[idx_label])
            desc_idx = local_idx_label[desc_id]
            
           
            if disc_id == -2: #if no split performed, do not add anything
                local_discriminatives += []
            else:   
                disc_idx = local_idx_label[disc_id]
                if isinstance(disc_idx, (list, np.ndarray)):
                    local_discriminatives += disc_idx.flatten().tolist() if isinstance(disc_idx, np.ndarray) else list(
                        disc_idx)
                else:
                    local_discriminatives += [disc_idx]


            local_descriptives += [desc_idx]

        local_candidates = local_descriptives + local_discriminatives

        self.X_candidates = X[local_candidates]
        self.y_candidates = y[local_candidates]

        self.discriminative_names = idx[local_discriminatives]
        self.descriptive_names = idx[local_descriptives]
        self.candidates_names = idx[local_candidates]

    def transform(self, X, distance_measure):
        """
        Transforms input data using the selected pivot instances.
        
        Computes distances from each input instance to each pivot instance,
        creating a new feature representation based on these distances.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.
        distance_measure : str or callable
            Distance measure to use (e.g., 'euclidean', 'cosine').
            
        Returns
        -------
        array-like of shape (n_samples, n_candidates)
            Distances from each instance to the pivot instances.
        """
        return pairwise_distances(X, self.X_candidates, metric=distance_measure)

    def get_candidates_names(self):
        """
        Returns the names/indices of candidate instances.
        
        These are the instances selected as potential split points,
        including both descriptive and discriminative instances.

        Returns
        -------
        array-like
            Names/indices of candidate instances.
        """
        return self.candidates_names

    def get_descriptive_names(self):
        """
        Returns the names/indices of descriptive instances.
        
        These are the instances that best represent each class (the medoids).

        Returns
        -------
        array-like
            Names/indices of descriptive instances.
        """
        return self.descriptive_names

    def get_discriminative_names(self):
        """
        Returns the names/indices of discriminative instances.
        
        These are the instances that best separate different classes.

        Returns
        -------
        array-like
            Names/indices of discriminative instances.
        """
        return self.discriminative_names
