import heapq

import numpy as np
import sklearn
from sklearn import tree
from sklearn.base import ClassifierMixin

from ruletree.stumps.classification.MultiplePivotTreeStumpClassifier import MultiplePivotTreeStumpClassifier
from ruletree.stumps.classification.PivotTreeStumpClassifier import PivotTreeStumpClassifier
from ruletree.tree.RuleTree import RuleTree
from ruletree.tree.RuleTreeNode import RuleTreeNode
from ruletree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from ruletree.utils.data_utils import calculate_mode, get_info_gain

from ruletree.utils.utils_decoding import configure_non_cat_split, configure_cat_split
from ruletree.utils.utils_decoding import set_node_children , simplify_decode
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

    def make_split(self, X: np.ndarray, y, idx: np.ndarray, sample_weight=None, **kwargs) -> tree:
        if self.stump_selection == 'random':
            stump = self._get_random_stump(X)
            
            if stump.__class__.__module__.split('.')[-1] in ['PivotTreeStumpClassifier','MultiplePivotTreeStumpClassifier']:
                if self.distance_matrix is None:
                    self.distance_matrix = pairwise_distances(X[idx], metric = self.distance_measure)
                stump.fit(X[idx], y[idx], distance_matrix=self.distance_matrix[idx][:,idx], idx=idx, 
                          distance_measure = self.distance_measure, sample_weight=None if sample_weight is None else sample_weight[idx]) 
            else:
                stump.fit(X[idx], y[idx], sample_weight=None if sample_weight is None else sample_weight[idx])
                
        elif self.stump_selection == 'best':
            clfs = []
            info_gains = []
            for _, stump in self._filter_types(X):
                stump = sklearn.clone(stump)
                
                if stump.__class__.__module__.split('.')[-1] in ['PivotTreeStumpClassifier','MultiplePivotTreeStumpClassifier']:
                    if self.distance_matrix is None:
                        self.distance_matrix = pairwise_distances(X[idx], metric = self.distance_measure)
                    stump.fit(X[idx], y[idx], distance_matrix=self.distance_matrix[idx][:,idx], idx=idx, 
                              distance_measure = self.distance_measure, sample_weight=None if sample_weight is None else sample_weight[idx]) 
                else:
                    stump.fit(X[idx], y[idx], sample_weight=None if sample_weight is None else sample_weight[idx])
            
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

    def fit(self, X: np.array, y: np.array=None, sample_weight=None, **kwargs):
        super().fit(X, y, sample_weight=sample_weight, **kwargs)

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
