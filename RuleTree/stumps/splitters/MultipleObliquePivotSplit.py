import itertools

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.tree import DecisionTreeClassifier
from inspect import signature

from RuleTree.stumps.splitters.ObliquePivotSplit import ObliquePivotSplit
from RuleTree.utils import MODEL_TYPE_CLU, MODEL_TYPE_REG, MODEL_TYPE_CLF
from RuleTree.utils.data_utils import get_info_gain


class MultipleObliquePivotSplit(ObliquePivotSplit):
    """
    MultipleObliquePivotSplit extends ObliquePivotSplit to find the best pair of pivot instances for splits.
    
    This class evaluates all possible pairs of pivot instances identified by the ObliquePivotSplit
    method and selects the pair that yields the highest information gain when used for splitting the data.
    
    Parameters
    ----------
    **kwargs : dict
        Additional parameters to pass to the parent ObliquePivotSplit class.
    
    Attributes
    ----------
    best_tup : array-like
        Best tuple of pivot instances for splitting.
    best_tup_name : array-like
        Names/indices of the best tuple of pivot instances.
    best_gain : float
        Information gain achieved by the best tuple.
    """
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.best_tup = None
        self.best_tup_name = None
        self.best_gain = -float('inf')

    def find_best_tuple(self, X, y, distance_measure='euclidean', sample_weight=None, check_input=True):
        """
        Find the best pair of pivot instances that maximizes information gain.
        
        Evaluates all possible pairs of candidate instances and selects the pair
        that yields the highest information gain when used for splitting.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.
        y : array-like of shape (n_samples,)
            Target values.
        distance_measure : str or callable, default='euclidean'
            Distance measure to use.
        sample_weight : array-like, optional
            Sample weights.
        check_input : bool, default=True
            Whether to validate input.
        """
        two_tuples = list(itertools.combinations(range(0, len(self.X_candidates)), 2))
    
        for tup in two_tuples:
            # Skip pairs from the same class
            if len(set(self.y_candidates[np.array(tup)])) == 1:
                continue
                
            disc = self.get_base_model_for_tuple_finding()
           
            p1, p2 = self.X_candidates[np.array(tup)]
            name_p1, name_p2 = self.candidates_names[np.array(tup)]

            # Calculate distances to both pivot instances
            dist_to_p0 = pairwise_distances(X, p1.reshape(1, -1), metric=distance_measure).flatten()
            dist_to_p1 = pairwise_distances(X, p2.reshape(1, -1), metric=distance_measure).flatten()

            # Create binary features: Is point closer to p1 or p2?
            dist_binary = np.where(dist_to_p0 < dist_to_p1, 0, 1).reshape(-1, 1)
            disc.fit(dist_binary, y)
            gain_disc = get_info_gain(disc)

            # Update if better gain found
            if gain_disc > self.best_gain:
                self.best_gain = gain_disc
                self.best_tup = self.X_candidates[np.array(tup)]
                self.best_tup_name = self.candidates_names[np.array(tup)]

    def fit(self, X, y, distance_matrix, distance_measure, idx,
            sample_weight=None, check_input=True):
        """
        Fit the MultipleObliquePivotSplit.
        
        First calls the parent ObliquePivotSplit fit method to identify candidates,
        then finds the best pair of pivot instances for splitting.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.
        y : array-like of shape (n_samples,)
            Target values.
        distance_matrix : array-like of shape (n_samples, n_samples)
            Pre-computed distance matrix between instances.
        distance_measure : str or callable
            Distance measure to use.
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
        super().fit(X, y, distance_matrix, distance_measure, idx, sample_weight=sample_weight, check_input=check_input)
        self.find_best_tuple(X, y, distance_measure=distance_measure, sample_weight=sample_weight,
                             check_input=check_input)

    def transform(self, X, distance_measure='euclidean'):
        """
        Transform input data using the best pair of pivot instances.
        
        For each input instance, determines whether it's closer to the first or
        second pivot instance in the best pair.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.
        distance_measure : str or callable, default='euclidean'
            Distance measure to use.
            
        Returns
        -------
        array-like of shape (n_samples, 1)
            Binary feature indicating whether each instance is closer to the first (0)
            or second (1) pivot instance.
        """
        dist_to_p0 = pairwise_distances(X, self.best_tup[0].reshape(1, -1), metric=distance_measure).flatten()
        dist_to_p1 = pairwise_distances(X, self.best_tup[1].reshape(1, -1), metric=distance_measure).flatten()
        dist_binary = np.where(dist_to_p0 < dist_to_p1, 0, 1).reshape(-1, 1)
        return dist_binary

    def get_best_tup_names(self):
        """
        Returns the names/indices of the best pair of pivot instances.
        
        Returns
        -------
        array-like
            Names/indices of the best pair of pivot instances.
        """
        return self.best_tup_name

    def get_base_model_for_tuple_finding(self):
        """
        Returns the appropriate base model for tuple finding based on the machine learning task.
        
        Filters the kwargs to include only valid parameters for the selected model.
        
        Returns
        -------
        model : estimator
            The machine learning model to use for finding the best tuple.
            
        Raises
        ------
        NotImplementedError
            If the ml_task is regression or clustering, which are not yet implemented.
        """
        if self.ml_task == MODEL_TYPE_CLF:
            valid_params = set(signature(DecisionTreeClassifier).parameters.keys())
            filtered_kwargs = {key: value for key, value in self.kwargs.items() if key in valid_params}
            return DecisionTreeClassifier(**filtered_kwargs)
        elif self.ml_task == MODEL_TYPE_REG:
            raise NotImplementedError("Regression is not implemented yet.")
        elif self.ml_task == MODEL_TYPE_CLU:
            raise NotImplementedError("Clustering is not implemented yet.")
