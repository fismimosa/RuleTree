import heapq

import numpy as np
import sklearn
from sklearn import tree
from sklearn.base import ClassifierMixin
import copy

from RuleTree.stumps.classification.MultiplePivotTreeStumpClassifier import MultiplePivotTreeStumpClassifier
from RuleTree.stumps.classification.PivotTreeStumpClassifier import PivotTreeStumpClassifier
from RuleTree.stumps.classification.ObliquePivotTreeStumpClassifier import ObliquePivotTreeStumpClassifier
from RuleTree.tree.RuleTree import RuleTree
from RuleTree.tree.RuleTreeNode import RuleTreeNode
from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from RuleTree.utils.data_utils import calculate_mode, get_info_gain

from RuleTree.utils.utils_decoding import configure_non_cat_split, configure_cat_split
from RuleTree.utils.utils_decoding import set_node_children , simplify_decode
from sklearn.metrics import pairwise_distances


class RuleTreeClassifier(RuleTree, ClassifierMixin):
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
        if base_stumps is None:
            base_stumps = DecisionTreeStumpClassifier(
                                max_depth=1,
                                criterion=criterion,
                                splitter=splitter,
                                min_samples_split=min_samples_split,
                                min_samples_leaf = min_samples_leaf,
                                min_weight_fraction_leaf=min_weight_fraction_leaf,
                                max_features=max_features,
                                random_state=random_state,
                                min_impurity_decrease=min_impurity_decrease,
                                class_weight=class_weight,
                                ccp_alpha=ccp_alpha,
                                monotonic_cst = monotonic_cst
            )

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


    def is_split_useless(self, X, clf: tree, idx: np.ndarray):
        labels = clf.apply(X[idx])

        return len(np.unique(labels)) == 1

    def check_additional_halting_condition(self, y, curr_idx: np.ndarray):
        return len(np.unique(y[curr_idx])) == 1  # only 1 target

    def queue_push(self, node: RuleTreeNode, idx: np.ndarray):
        heapq.heappush(self.queue, (len(node.node_id), next(self.tiebreaker), idx, node))

    def make_split(self, X: np.ndarray, y, idx: np.ndarray, medoids_index = None, sample_weight=None, **kwargs) -> tree:
        
        pivots_list = ['PivotTreeStumpClassifier',
                        'MultiplePivotTreeStumpClassifier',
                        'ObliquePivotTreeStumpClassifier',
                        'MultipleObliquePivotTreeStumpClassifier']
        
    
        if self.stump_selection == 'random':
            stump = self._get_random_stump(X)
        
            if stump.__class__.__module__.split('.')[-1] in pivots_list:
                
                
                stump.fit(X[idx], y[idx], distance_matrix=self.distance_matrix[idx][:,idx], idx=idx,
                         
                          distance_measure = self.distance_measure, sample_weight=None if sample_weight is None else sample_weight[idx]) 
            else:
                stump.fit(X=X,
                          y=y,
                          idx=idx,
                          context = self,
                          sample_weight=None if sample_weight is None else sample_weight[idx])
                
        elif self.stump_selection == 'best':
            clfs = []
            info_gains = []
            for _, stump in self._filter_types(X):
                stump = sklearn.clone(stump)
                
                if stump.__class__.__module__.split('.')[-1] in pivots_list:
                    
                    stump.fit(X=X[idx],
                              y=y[idx],
                              distance_matrix=self.distance_matrix[idx][:,idx],
                              idx=idx,
                              distance_measure=self.distance_measure,
                              sample_weight=None if sample_weight is None else sample_weight[idx])
                else:
                    stump.fit(X=X,
                              y=y,
                              idx=idx,
                              context=self,
                              sample_weight=None if sample_weight is None else sample_weight[idx])
            
                gain = get_info_gain(stump)
                info_gains.append(gain)
                
                clfs.append(stump)

            stump = clfs[np.argmax(info_gains)]
        else:
            raise TypeError('Unknown stump selection method')

        return stump

    def prepare_node(self, y: np.ndarray, idx: np.ndarray, node_id: str) -> RuleTreeNode:
        prediction = calculate_mode(y[idx])
        predict_proba = np.zeros((len(self.classes_), ))
        for i, classe in enumerate(self.classes_):
            predict_proba[i] = sum(np.where(y[idx] == classe, 1, 0)) / len(y[idx])


        return RuleTreeNode(
            node_id=node_id,
            prediction=prediction,
            prediction_probability=predict_proba,
            classes = self.classes_,
            parent=None,
            stump=None,
            node_l=None,
            node_r=None,
            samples=len(y[idx]),
        )

    def compute_medoids(self, X: np.ndarray, y, idx: np.ndarray, **kwargs):
        if self.distance_measure is not None:
            medoids = []
            sub_matrix = None
            for label in set(y[idx]):
                idx_local_label = np.where(y[idx] == label)[0]
                idx_label = idx[idx_local_label]
                X_class_points = X[idx_label]
                
                if self.distance_matrix is not None:
                    sub_matrix = self.distance_matrix[idx_label][:,idx_label]
                else:
                    sub_matrix = pairwise_distances(X_class_points, metric=self.distance_measure)
                total_distances = sub_matrix.sum(axis=1)
                medoid_index = idx_label[total_distances.argmin()]
                medoids += [medoid_index]
                
            return medoids
             
    def fit(self, X: np.array, y: np.array = None, sample_weight=None, **kwargs):
        # Check and initialize the distance matrix if needed
        if self.distance_matrix is None and self.base_stumps is not None:
            base_stumps = self.base_stumps if isinstance(self.base_stumps, list) else [self.base_stumps]
            for stump in base_stumps:
                # Check if the class name matches the specified list
                if stump.__class__.__module__.split('.')[-1] in [
                    'PivotTreeStumpClassifier',
                    'MultiplePivotTreeStumpClassifier',
                    'ObliquePivotTreeStumpClassifier',
                    'MultipleObliquePivotTreeStumpClassifier'
                ]:
                    # Compute the distance matrix
                    self.distance_matrix = pairwise_distances(X, metric=self.distance_measure)
                    #print(X[0][0])
                    #print('compute dist')
                    #print(self.distance_matrix.shape)
                    
                
                    break  # Distance matrix is initialized, no need to continue
    
        
        super().fit(X, y, sample_weight=sample_weight, **kwargs)
        if self.distance_matrix is not None:
            self.distance_matrix = None #remove to save space when training many estimators

        return self


    def predict_proba(self, X: np.ndarray):
        labels, leaves, proba = self._predict(X, self.root)

        return proba


    def _predict(self, X: np.ndarray, current_node: RuleTreeNode):
        if current_node.is_leaf():
            n = len(X)
            return np.array([current_node.prediction] * n), \
                np.array([current_node.node_id] * n), \
                np.zeros((len(X), len(self.classes_)), dtype=float) + current_node.prediction_probability

        else:
            labels, leaves, proba = (
                np.full(len(X), fill_value=-1,
                        dtype=object if type(current_node.prediction) is str else type(current_node.prediction)),
                np.zeros(len(X), dtype=object),
                np.ones((len(X), len(self.classes_)), dtype=float) * -1
            )

            clf = current_node.stump
            labels_clf = clf.apply(X)
            X_l, X_r = X[labels_clf == 1], X[labels_clf == 2]
            if X_l.shape[0] != 0:
                labels[labels_clf == 1], leaves[labels_clf == 1], proba[labels_clf == 1] = self._predict(X_l,
                                                                                                         current_node.node_l)
            if X_r.shape[0] != 0:
                labels[labels_clf == 2], leaves[labels_clf == 2], proba[labels_clf == 2] = self._predict(X_r,
                                                                                                         current_node.node_r)

            return labels, leaves, proba
        
        
    def get_pivots(self, current_node=None, pivot_dicts=None):
        
        
   
        stump_split_map = {
            'PivotTreeStumpClassifier': 'pivot_split',
            'MultiplePivotTreeStumpClassifier': 'multi_pivot_split',
            'ObliquePivotTreeStumpClassifier': 'obl_pivot_split',
            'MultipleObliquePivotTreeStumpClassifier' : 'multi_oblique_pivot_split',
        
            'ObliqueDecisionTreeStumpClassifier' : 'oblique_split',
            'DecisionTreeStumpClassifier' : 'univariate_split'
        }
        
        
        # Initialize pivot_dicts if not provided
        if pivot_dicts is None:
            pivot_dicts = {}
        
        # Start from root if current_node is not provided
        if current_node is None:
            current_node = self.root
            
        
        # Process current node
        if current_node.stump is not None:
            
            stump_name = current_node.stump.__class__.__module__.split('.')[-1]
            used = current_node.stump.feature_original[0]
            if stump_split_map[stump_name] == 'pivot_split':
                used = [int(used)]
            if stump_split_map[stump_name] == 'multi_pivot_split':
                used = list(used)
            if stump_split_map[stump_name] == 'multi_oblique_pivot_split':
                used = list(used)
            if stump_split_map[stump_name] == 'obl_pivot_split':
                used = [int(x) for x in used]
            
            
            if stump_name in stump_split_map:
                if stump_split_map[stump_name] in ['oblique_split', 'univariate_split']:
                    pivot_dicts[current_node.node_id]  = {'descriptives' : current_node.medoids_index}
                    
                    
                if stump_split_map[stump_name] in ['pivot_split','multi_pivot_split','multi_oblique_pivot_split','obl_pivot_split']:
                    split_obj = getattr(current_node.stump, stump_split_map[stump_name])
                    if not current_node.is_leaf():
                        pivot_dicts[current_node.node_id] = {
                                'discriminatives': split_obj.get_discriminative_names(),
                                'descriptives': split_obj.get_descriptive_names(),
                                'candidates': split_obj.get_candidates_names(),
                                'used' : used
                            }
                    else:
                        pivot_dicts[current_node.node_id]  = {'descriptives' : split_obj.get_descriptive_names()}
                        

        
        else:
            if current_node.is_leaf():
                pivot_dicts[current_node.node_id]  = {'descriptives' : current_node.medoids_index}
                
        
        # Recurse into child nodes if they exist
        if current_node.node_l:
            self.get_pivots(current_node.node_l, pivot_dicts)
            self.get_pivots(current_node.node_r, pivot_dicts)
        
        return pivot_dicts

        
        
        
    def _get_stumps_base_class(self):
        return ClassifierMixin

    def _get_prediction_probas(self, current_node = None, probas=None):
        if probas is None:
            probas = []
            
        if current_node is None:
            current_node = self.root
        
        if current_node.prediction is not None:
            probas.append(current_node.prediction_probability)
           
        if current_node.node_l:
            self._get_prediction_probas(current_node.node_l, probas)
            self._get_prediction_probas(current_node.node_r, probas)
        
        return probas

    def local_interpretation(self, X, joint_contribution = False):
        leaves, paths, leaf_to_path, values = super().local_interpretation(X = X,
                                                                           joint_contribution = joint_contribution)
        normalizer = values.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        values /= normalizer

        biases = np.tile(values[paths[0][0]], (X.shape[0], 1))
        line_shape = (X.shape[1], self.n_classes_)
        
        return super().eval_contributions(
                                        leaves=leaves,
                                        paths=paths,
                                        leaf_to_path=leaf_to_path,
                                        values=values,
                                        biases=biases,
                                        line_shape=line_shape,
                                        joint_contribution=joint_contribution
                                    )
   

    
    def get_balanced_stumps(self, current_node=None, stumps=None, p=0.2):
        if stumps is None:
            stumps = {}
        if current_node is None:
            current_node = self.root
            
        if not current_node.is_leaf():
            if current_node.balance_score > p:
                stumps[current_node.node_id] = (current_node, current_node.balance_score)
                self.get_balanced_stumps(current_node=current_node.node_l, stumps=stumps, p=p)
                self.get_balanced_stumps(current_node=current_node.node_r, stumps=stumps, p=p)
                
        return stumps
    
    def stumps_to_trees(self, balanced_nodes):
        trees = {}
      
        for k, v in balanced_nodes.items():
            rt = self.__class__()
            rt.classes_ = self.classes_
            
            node = copy.deepcopy(v[0])
            node_l = copy.deepcopy(node.node_l)
            node_r = copy.deepcopy(node.node_r)
                        
            feat = tuple(node.stump.feature_original[0])  # Ensure it's a tuple

            thr = (node.stump.threshold_original[0],)
                        
            #print(feat)
            #print(thr)
            
            
            rt.root = node
            rt.root.node_l = node_l
            rt.root_node_r = node_r
            
            rt.root.node_l.make_leaf()
            rt.root.node_r.make_leaf()
            
            rt.root.node_id, rt.root.node_l.node_id, rt.root.node_r.node_id = 'R', 'Rl', 'Rr'
            
            
            #print('ids', rt.root.node_id, rt.root.node_l.node_id)
            
            trees[(feat, thr)] = rt
            
        return trees
            
        
        
    
    

        
    
    @classmethod
    def complete_tree(cls, node, X, y, n_classes_):
        classes_ = [i for i in range(n_classes_)]
        node.prediction = calculate_mode(y)
        node.prediction_probability = np.zeros((len(classes_), ))
        node.samples = len(y)
        
        
        for i, classe in enumerate(classes_):
            node.prediction_probability[i] = np.sum(np.where(y == classe, 1, 0)) / len(y)

        if not node.is_leaf():            
            labels_clf = node.stump.apply(X)
            X_l, X_r = X[labels_clf == 1], X[labels_clf == 2]
            y_l, y_r = y[labels_clf == 1], y[labels_clf == 2]         
            if X_l.shape[0] != 0:
                cls.complete_tree(node.node_l, X_l, y_l, n_classes_)
            if X_r.shape[0] != 0:
                cls.complete_tree(node.node_r, X_r, y_r, n_classes_)

            
        
    @classmethod    
    def decode_ruletree(cls, vector, n_features_in_, n_classes_, n_outputs_, 
                        numerical_idxs=None, categorical_idxs=None):
        
        idx_to_node = super().decode_ruletree(vector)
        
        for index in range(len(vector[0])):
            #if leaf
            if vector[0][index] == -1:
                idx_to_node[index].prediction = vector[1][index]
            else:
                clf = DecisionTreeStumpClassifier() ##add kwargs in the function
                clf.numerical = numerical_idxs
                clf.categorical = categorical_idxs
                if isinstance(vector[1][index], str):
                    clf = configure_cat_split(clf, vector[0][index], vector[1][index])
                else:
                    clf = configure_non_cat_split(clf, vector, index, 
                                               n_features_in_, n_classes_, n_outputs_)
                    
                idx_to_node[index].stump = clf
                set_node_children(idx_to_node, index, vector)
                
        
        rule_tree = RuleTreeClassifier()
        rule_tree.classes_ = [i for i in range(n_classes_)]
        simplify_decode(idx_to_node[0])
        rule_tree.root = idx_to_node[0]
        return rule_tree
                
                
        
            
    @classmethod
    def _decode_old(cls, vector, n_features_in_, n_classes_, n_outputs_, 
                        numerical_idxs=None, categorical_idxs=None, criterion=None):
        
        idx_to_node = super().decode_ruletree(vector, n_features_in_, n_classes_, n_outputs_, 
                                              numerical_idxs, categorical_idxs, criterion)
        
    
        for index in range(len(vector[0])):
            if vector[0][index] == -1:
                idx_to_node[index].prediction = vector[1][index]
            else:
                clf = DecisionTreeStumpClassifier(
                                        criterion=criterion)
                
                clf = DecisionTreeStumpClassifier()
        
                if numerical_idxs is not None:
                   clf.numerical = numerical_idxs
        
                if categorical_idxs is not None:
                   clf.categorical = categorical_idxs
                            
                if isinstance(vector[1][index], str):
                    configure_cat_split(clf, vector[0][index], vector[1][index])
                else:
                    configure_non_cat_split(clf, vector, index, 
                                               n_features_in_, n_classes_, n_outputs_)
                idx_to_node[index].stump = clf
                set_node_children(idx_to_node, index, vector)
                
                print(clf)
                
        rule_tree = RuleTreeClassifier()
        simplify_decode(idx_to_node[0])
        rule_tree.root = idx_to_node[0]
        return rule_tree
