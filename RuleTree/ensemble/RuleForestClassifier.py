from random import random
import numpy as np
import sklearn.base
from sklearn.ensemble import BaggingClassifier
from RuleTree.utils.data_utils import _iterative_mean

from RuleTree import RuleTreeClassifier
from RuleTree.base.RuleTreeBase import RuleTreeBase
from sklearn.base import ClassifierMixin

from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier

class RuleForestClassifier(BaggingClassifier, RuleTreeBase):
    def __init__(self,
                 n_estimators=100,
                 criterion='gini',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 min_impurity_decrease=0.0,
                 max_leaf_nodes=float("inf"),
                 class_weight=None,
                 ccp_alpha=0.0,
                 prune_useless_leaves=False,
                 splitter='best',
                 *,
                 max_samples=None,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features = True,
                 oob_score=False,
                 warm_start=False,
                 custom_estimator:sklearn.base.ClassifierMixin=None,
                 n_jobs=None,
                 random_state=None,
                 base_stumps = None,
                 distance_matrix = None,
                 distance_measure = None,
                 stump_selection = 'best',
                 verbose=0):
        
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_leaf_nodes = max_leaf_nodes
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.prune_useless_leaves = prune_useless_leaves
        self.splitter = splitter
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.custom_estimator = custom_estimator
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.base_stumps = base_stumps
        self.distance_matrix = distance_matrix
        self.distance_measure = distance_measure
        self.stump_selection= stump_selection
    
               

    def fit(self, X: np.ndarray, y:np.ndarray, sample_weight=None, **kwargs):
        if self.max_features is None:
            self.max_features = X.shape[1]
            

        if type(self.max_features) is str:
            if self.max_features == "sqrt":
                self.max_features = int(np.sqrt(X.shape[1]))
            elif self.max_features == "log2":
                self.max_features = int(np.log2(X.shape[1]))

        base_estimator = RuleTreeClassifier if self.custom_estimator is None else self.custom_estimator
        splitter = .5 if self.splitter == 'hybrid_forest' else self.splitter
        
        if type(splitter) is float:
            base_estimator = RuleTreeClassifier_choosing_splitter_randomly
            
        if self.base_stumps is None:
            self.base_stumps = [DecisionTreeStumpClassifier(
                                max_depth=1,
                                criterion=self.criterion,
                                splitter=self.splitter,
                                min_samples_split=self.min_samples_split,
                                min_samples_leaf = self.min_samples_leaf,
                                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                max_features=self.max_features,
                                random_state=self.random_state,
                                min_impurity_decrease=self.min_impurity_decrease,
                                class_weight=self.class_weight,
                                ccp_alpha=self.ccp_alpha,
                              #  monotonic_cst = self.monotonic_cst
                                )]
        
        else:
            for stump in self.base_stumps:
                if stump.__class__.__module__.split('.')[-1] in [
                    'PivotTreeStumpClassifier',
                    'MultiplePivotTreeStumpClassifier',
                    'ObliquePivotTreeStumpClassifier',
                    'MultipleObliquePivotTreeStumpClassifier'
                ]:
                    
                    base_estimator = ForestEstimatorPivotClassifier
                    break
                
            
     
        super().__init__(estimator=base_estimator(criterion=self.criterion,
                                                  max_depth=self.max_depth,
                                                  min_samples_split=self.min_samples_split,
                                                  min_samples_leaf=self.min_samples_leaf,
                                                  min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                                  min_impurity_decrease=self.min_impurity_decrease,
                                                  random_state=self.random_state,
                                                  max_leaf_nodes=self.max_leaf_nodes,
                                                  class_weight=self.class_weight,
                                                  ccp_alpha=self.ccp_alpha,
                                                  prune_useless_leaves=self.prune_useless_leaves,
                                                  splitter=self.splitter,
                                                  base_stumps = self.base_stumps,
                                                  distance_measure = self.distance_measure,
                                                  distance_matrix = self.distance_matrix,
                                                  stump_selection= self.stump_selection
                        
                                                  ),
                         n_estimators=self.n_estimators,
                         max_samples=X.shape[0] if self.max_samples is None else self.max_samples,
                         max_features=self.max_features,
                         bootstrap=self.bootstrap,
                         bootstrap_features=self.bootstrap_features,
                         oob_score=self.oob_score,
                         warm_start=self.warm_start,
                         n_jobs=self.n_jobs,
                         random_state=self.random_state,
                         verbose=self.verbose)

        return super().fit(X, y, sample_weight=sample_weight, **kwargs)
     
    def local_interpretation(self, X, joint_contribution = False):
        
        if joint_contribution:
            biases = []
            contributions = []
            predictions = []
            
            for tree in self.estimators_:
                pred, bias, contribution = tree.local_interpretation(X, joint_contribution=joint_contribution)
                biases.append(bias)
                contributions.append(contribution)
                predictions.append(pred)
                
            total_contributions = []
            
            for i in range(len(X)):
                contr = {}
                for j, dct in enumerate(contributions):
                    for k in set(dct[i]).union(set(contr.keys())):
                        contr[k] = (contr.get(k, 0)*j + dct[i].get(k,0) ) / (j+1)

                total_contributions.append(contr)    
                
            for i, item in enumerate(contribution):
                total_contributions[i]
                sm = sum([v for v in contribution[i].values()])
                    

            
            return (np.mean(predictions, axis=0), np.mean(biases, axis=0),
                total_contributions)
        else:
            mean_pred = None
            mean_bias = None
            mean_contribution = None
            
            for i, tree in enumerate(self.estimators_):
                pred, bias, contribution = tree.local_interpretation(X)
                
                if i < 1: # first iteration
                    mean_bias = bias
                    mean_contribution = contribution
                    mean_pred = pred
                else:
                    mean_bias = _iterative_mean(i, mean_bias, bias)
                    mean_contribution = _iterative_mean(i, mean_contribution, contribution)
                    mean_pred = _iterative_mean(i, mean_pred, pred)

            return mean_pred, mean_bias, mean_contribution

        

class RuleTreeClassifier_choosing_splitter_randomly(RuleTreeClassifier):
    def __init__(self, splitter, **kwargs):
        if random() < splitter:
            if random() < splitter:
                splitter = 'random'
            else:
                splitter = 'best'
        kwargs["splitter"] = splitter
        super().__init__(**kwargs)
        
class ForestEstimatorPivotClassifier(RuleTreeClassifier):
    def __init__(self,
                 max_leaf_nodes=float('inf'),
                 min_samples_split=2,
                 max_depth=float('inf'),
                 prune_useless_leaves=False,
                 base_stumps: ClassifierMixin | list = None,
                 stump_selection: str = 'random',
                 random_state=None,

                 criterion='gini',
                 splitter='best',
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features=None,
                 min_impurity_decrease=0.0,
                 class_weight=None,
                 ccp_alpha=0.0,
                 monotonic_cst=None,
                 distance_matrix = None,
                 distance_measure = None
                 
                 ):
        
        super().__init__(max_leaf_nodes=max_leaf_nodes,
                         min_samples_split=min_samples_split,
                         max_depth=max_depth,
                         prune_useless_leaves=prune_useless_leaves,
                         base_stumps=base_stumps,
                         stump_selection=stump_selection,
                         random_state=random_state)

        self.max_depth = max_depth
        self.criterion = criterion
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst
        self.distance_matrix = distance_matrix    
        self.distance_measure = distance_measure

    
    def fit(self, X: np.array, y: np.array=None, **kwargs):
        super().fit(X, y, **kwargs)
