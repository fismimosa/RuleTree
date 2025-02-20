import heapq
import importlib
import json
import random
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from itertools import count

import numpy as np
import sklearn
from sklearn import tree
from tempfile312 import TemporaryDirectory

from RuleTree.base.RuleTreeBase import RuleTreeBase
from RuleTree.tree.RuleTreeNode import RuleTreeNode
from RuleTree.base.RuleTreeBaseStump import RuleTreeBaseStump
from RuleTree.utils.data_utils import json_NumpyEncoder
from RuleTree.utils.define import DATA_TYPE_IMAGE, DATA_TYPE_TABULAR, DATA_TYPE_TS


class RuleTree(RuleTreeBase, ABC):
    def __init__(self,
                 max_leaf_nodes,
                 min_samples_split,
                 max_depth,
                 prune_useless_leaves,
                 base_stumps, #isinstance of RuleTreeBaseStump, list of instances of RuleTreeBaseStump,
                                # list of tuple (probability, RuleTreeBaseStump) where sum(probabilities)==1
                 stump_selection, # ['random', 'best']
                 random_state,
                 ):
        self.tiebreaker = count()
        self.root:RuleTreeNode = None
        self.queue = list()
        self.classes_ = None
        self.n_classes_ = None

        self.max_leaf_nodes = float("inf") if max_leaf_nodes is None else max_leaf_nodes
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.prune_useless_leaves = prune_useless_leaves

        self.base_stumps = base_stumps
        self.stump_selection = stump_selection

        self.random_state = random_state
        random.seed(random_state)

    def _set_stump(self):
        # base stump type checking
        class_to_check = self._get_stumps_base_class()
        _base_stump = []
        _p = []
        if isinstance(self.base_stumps, RuleTreeBaseStump):
            assert isinstance(self.base_stumps, class_to_check)
            _base_stump.append(self.base_stumps)
            _p.append(1.)
        elif type(self.base_stumps) == list:
            assert len(self.base_stumps) > 0
            _equal_p = 1 / len(self.base_stumps)
            for el in self.base_stumps:
                if isinstance(el, Sequence):
                    assert isinstance(el[1], RuleTreeBaseStump)
                    assert isinstance(el[1], class_to_check)
                    _p.append(el[0])
                    _base_stump.append(el[1])
                else:
                    assert isinstance(el, RuleTreeBaseStump)
                    assert isinstance(el, class_to_check)
                    _p.append(_equal_p)
                    _base_stump.append(el)

        assert sum(_p) == 1.
        self.base_stumps = [(p, stump) for p, stump in zip(np.cumsum(_p), _base_stump)]

    def _get_random_stump(self, X):
        val = random.random()

        supported_stumps = self._filter_types(X)

        for p, clf in supported_stumps:
            if val <= p:
                return sklearn.clone(clf)

    def _filter_types(self, X) -> list[tuple[float, RuleTreeBaseStump]]:
        if len(X.shape) == 2:
            data_type = DATA_TYPE_TABULAR
        elif len(X.shape) == 3:
            data_type = DATA_TYPE_TS
        elif len(X.shape) == 4:
            data_type = DATA_TYPE_IMAGE
        else:
            raise TypeError(f"Invalid data type for shape {X.shape}")

        compatible_stumps = [(p, x) for p, x in self.base_stumps if x.supports(data_type)]

        if len(compatible_stumps) == 0:
            raise TypeError(f"No compatible stumps found for "
                            f"shape {X.shape}.\r\n{[x.__name__ for _, x in self.base_stumps]}")

        p_total = sum([x for x, _ in compatible_stumps])
        if p_total < 1:
            compatible_stumps = [(p/p_total, x) for p, x in compatible_stumps]

        return compatible_stumps
    


    @abstractmethod
    def compute_medoids(self, X: np.ndarray, y, idx: np.ndarray, **kwargs):
        pass
    
    def fit(self, X: np.array, y: np.array = None, **kwargs):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features = X.shape[1]
        self._set_stump()

        idx = np.arange(X.shape[0])

        self.root = self.prepare_node(y, idx, "R")

        self.queue_push(self.root, idx)

        nbr_curr_nodes = 0
        while len(self.queue) > 0 and nbr_curr_nodes + len(self.queue) < self.max_leaf_nodes:
            idx, current_node = self.queue_pop()
            
            
            if len(idx) < self.min_samples_split:
                self.make_leaf(current_node)
                current_node.medoids_index = self.compute_medoids(X, y, idx=idx, **kwargs)
                nbr_curr_nodes += 1
                continue

            if nbr_curr_nodes + len(self.queue) + 1 >= self.max_leaf_nodes:
                self.make_leaf(current_node)
                current_node.medoids_index = self.compute_medoids(X, y, idx=idx, **kwargs)
                nbr_curr_nodes += 1
                continue

            if self.max_depth is not None and current_node.get_depth() >= self.max_depth:
                self.make_leaf(current_node)
                current_node.medoids_index = self.compute_medoids(X, y, idx=idx, **kwargs)
                nbr_curr_nodes += 1
                continue

            if self.check_additional_halting_condition(y=y, curr_idx=idx):
                self.make_leaf(current_node)
                current_node.medoids_index = self.compute_medoids(X, y, idx=idx, **kwargs)
                nbr_curr_nodes += 1
                continue

            clf = self.make_split(X, y, idx=idx, **kwargs)
            labels = clf.apply(X[idx])
            
           
            name_clf = clf.__class__.__module__.split('.')[-1]
            #print(name_clf)
            
            if name_clf in ['ObliqueDecisionTreeStumpClassifier',
                            'DecisionTreeStumpClassifier']:
                current_node.medoids_index = self.compute_medoids(X, y, idx=idx, **kwargs)
                
                
            global_labels = clf.apply(X)
            current_node.balance_score_global = (np.min(np.unique(global_labels, return_counts= True)[1]) / global_labels.shape[0])
            current_node.balance_score = current_node.balance_score_global

            if self.is_split_useless(X=X, clf=clf, idx=idx):
                self.make_leaf(current_node)
                current_node.medoids_index = self.compute_medoids(X, y, idx=idx, **kwargs)
                nbr_curr_nodes += 1
                continue

            idx_l, idx_r = idx[labels == 1], idx[labels == 2]

            current_node.set_stump(clf)
            current_node.node_l = self.prepare_node(y, idx_l, current_node.node_id + "l", )
            current_node.node_r = self.prepare_node(y, idx_r, current_node.node_id + "r", )
            current_node.node_l.parent, current_node.node_r.parent = current_node, current_node
            
            
            #current_node.balance_score = (np.min(np.unique(labels, return_counts= True)[1]) / labels.shape[0])
            
            #global_labels = clf.apply(X)
            #current_node.balance_score_global = (np.min(np.unique(global_labels, return_counts= True)[1]) / global_labels.shape[0])
           
            #current_node.balance_score = (np.min(np.unique(global_labels, return_counts= True)[1]) / global_labels.shape[0])
           
            
            

            self.queue_push(current_node.node_l, idx_l)
            self.queue_push(current_node.node_r, idx_r)

        if self.prune_useless_leaves:
            self.root = self.root.simplify()

        self._post_fit_fix()

        return self

    def predict(self, X: np.ndarray):
        labels, leaves, proba = self._predict(X, self.root)

        return labels

    def apply(self, X: np.ndarray):
        labels, leaves, proba = self._predict(X, self.root)

        return leaves

    def predict_proba(self, X: np.ndarray):
        labels, leaves, proba = self._predict(X, self.root)
        proba_matrix = np.zeros((X.shape[0], self.n_classes_))
        for classe in self.classes_:
            proba_matrix[labels == classe, self.classes_ == classe] = proba[labels == classe]

        return proba_matrix

    def _predict(self, X: np.ndarray, current_node: RuleTreeNode):
        if current_node.is_leaf():
            n = len(X)
            return np.array([current_node.prediction] * n), \
                np.array([current_node.node_id] * n), \
                np.array([current_node.prediction_probability] * n)

        else:
            labels, leaves, proba = (
                np.full(len(X), fill_value=-1,
                        dtype=object if type(current_node.prediction) is str else type(current_node.prediction)),
                np.zeros(len(X), dtype=object),
                np.ones(len(X), dtype=float) * -1
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

    def get_rules(self, columns_names = None):
        return self.root.get_rule(columns_names = columns_names)

    def make_leaf(self, node: RuleTreeNode) -> RuleTreeNode:
        return node

    def queue_pop(self):
        el = heapq.heappop(self.queue)
        return el[-2:]

    def check_additional_halting_condition(self, y, curr_idx: np.ndarray):
        return False

    def _post_fit_fix(self):
        return

    @abstractmethod
    def make_split(self, X: np.ndarray, y, idx: np.ndarray, **kwargs) -> tree:
        pass

    @abstractmethod
    def prepare_node(self, y: np.ndarray, idx: np.ndarray, node_id: str) -> RuleTreeNode:
        pass

    @abstractmethod
    def queue_push(self, node: RuleTreeNode, idx: np.ndarray):
        pass

    @abstractmethod
    def is_split_useless(self, X: np.ndarray, clf: tree, idx: np.ndarray):
        pass

    @abstractmethod
    def _get_stumps_base_class(self):
        return RuleTreeBaseStump

    def _get_tree_paths(self, current_node = None):
        if current_node.is_leaf():
            return [[current_node.node_id]]
        
        if current_node is None:
            current_node = self.root
                

        if current_node.node_l:  
            left_paths = self._get_tree_paths(current_node.node_l)
            right_paths = self._get_tree_paths(current_node.node_r)
            
            for path in left_paths:
                path.append(current_node.node_id) 
                
            for path in right_paths:
                path.append(current_node.node_id)
                
            paths = left_paths + right_paths
            
        return paths
    
    def _node_dict(self, current_node=None, d=None, i=0):
        if d is None:
            d = {}
            
        if current_node is None:
            current_node = self.root
    
        # Assign an index to the current node
        if current_node.node_id not in d:
            d[current_node.node_id] = i
            i += 1
    
        if not current_node.is_leaf():
            i = self._node_dict(current_node.node_l, d, i)[-1]
            i = self._node_dict(current_node.node_r, d, i)[-1]
    
        return d, i

    @abstractmethod
    def _get_prediction_probas(self, current_node, probas=None):
        pass

    def _tree_value(self):
        probas = self._get_prediction_probas(self.root)
        if isinstance(probas[0], (list, np.ndarray)):
            return np.array(probas).reshape(-1, 1, len(probas[0]))
        else:
            return np.array(probas).reshape(-1, 1, 1)
            
    def _tree_feature(self, current_node = None, feats=None):
        if feats is None:
            feats = []
            
        if current_node is None:
            current_node = self.root
        
        if current_node.is_leaf():
            feats.append(-2)
        else:
            feats.append(current_node.stump.feature_original[0])
            self._tree_feature(current_node.node_l, feats = feats)
            self._tree_feature(current_node.node_r, feats = feats)
        
        return feats
        

    def _compute_importances(self, current_node=None, importances=None):
       if importances is None:
           importances = np.zeros(self.n_features, dtype=np.float64)

       if current_node is None:
           current_node = self.root

       if not current_node.is_leaf():
           #following the implementation of https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_tree.pyx#L1251
           feature = current_node.stump.feature_original[0]
           
           imp_parent, imp_child_l, imp_child_r = current_node.stump.tree_.impurity
           n_parent, n_child_l, n_child_r = current_node.stump.tree_.weighted_n_node_samples
           
           info_gain = (
               n_parent * imp_parent
               - n_child_l * imp_child_l
               - n_child_r * imp_child_r
           )
           
           info_gain 
           
           importances[feature] += info_gain

           # Recur for left and right children
           self._compute_importances(current_node.node_l, importances)
           self._compute_importances(current_node.node_r, importances)

       return importances 
   
    def compute_feature_importances(self, normalize=True):
        
        importances = self._compute_importances()
        root_weighted_samples = self.root.stump.tree_.weighted_n_node_samples[0]
        importances /= root_weighted_samples
        
        if normalize:
            total = np.sum(importances)
            if total > 0.0:
                importances /= total

        return importances
        
    def local_interpretation(self, X, joint_contribution = False):
        
        node_dict = self._node_dict()[0] #-> Turn 'R': 0, 'Rl' : 1, 'Rr' :2 and so on

        leaves = np.array([node_dict[x] for x in self.apply(X)])
        paths = [[node_dict[x] for x in path] for path in self._get_tree_paths(self.root)]
        
        for path in paths:
            path.reverse()
        
        leaf_to_path = {}
        
        for path in paths:
            leaf_to_path[path[-1]] = path
            
        values = self._tree_value().squeeze(axis = 1)
        
        return leaves, paths, leaf_to_path, values
        
    
    def eval_contributions(self, 
                           leaves, 
                           paths, 
                           leaf_to_path, 
                           values, 
                           biases,
                           line_shape,
                           joint_contribution = False):
        
        direct_prediction = values[leaves]
        values_list = list(values)
        feature_index = list(self._tree_feature())
        
        contributions = []
        if joint_contribution:
            for row, leaf in enumerate(leaves):
                path = leaf_to_path[leaf]
                
                
                path_features = set()
                contributions.append({})
                for i in range(len(path) - 1):
                    path_features.add(feature_index[path[i]])
                    contrib = values_list[path[i+1]] - \
                             values_list[path[i]]
                    #path_features.sort()
                    contributions[row][tuple(sorted(path_features))] = \
                        contributions[row].get(tuple(sorted(path_features)), 0) + contrib
            return direct_prediction, biases, contributions
        
        else:
            unique_leaves = np.unique(leaves)
            unique_contributions = {}
            
            for row, leaf in enumerate(unique_leaves):
                for path in paths:
                    if leaf == path[-1]:
                        break
                
                contribs = np.zeros(line_shape)
                for i in range(len(path) - 1):
                    
                    contrib = values_list[path[i+1]] - \
                             values_list[path[i]]
                    contribs[feature_index[path[i]]] += contrib
                unique_contributions[leaf] = contribs
                
            for row, leaf in enumerate(leaves):
                contributions.append(unique_contributions[leaf])

            return direct_prediction, biases, np.array(contributions)


    
    
    
    @classmethod
    def print_rules(cls, rules: dict, columns_names: list = None, ndigits=2, indent: int = 0, ):
        #columns_names now included in stump get_rule(), should we keep it here?
        #ndigits/float precision included in stump get_rule(), should we keep it here?
      
        indentation = "".join(["|   " for _ in range(indent)])
        
        if rules["is_leaf"]:
            pred = rules['prediction']
            print(f"{indentation} output: "
                  f"{pred if type(pred) in [np.str_, np.string_, str] else round(pred, ndigits=ndigits)}")
            
        else:
            print(f"{indentation}|--- {rules['textual_rule']}")
            cls.print_rules(rules=rules['left_node'], columns_names=columns_names, indent=indent + 1)
            print(f"{indentation}|--- {rules['not_textual_rule']}")
            cls.print_rules(rules=rules['right_node'], columns_names=columns_names, indent=indent + 1)
    
    
    @classmethod
    def print_rules_old(cls, rules: dict, columns_names: list = None, ndigits=2, indent: int = 0, ):
        names = lambda x: f"X_{x}"
        names_pivots = lambda x: f"P_{x}"

        if columns_names is not None:
            names = lambda x: columns_names[x]

        indentation = "".join(["|   " for _ in range(indent)])

        if rules["is_leaf"]:
            pred = rules['prediction']

            print(f"{indentation} output: "
                  f"{pred if type(pred) in [np.str_, np.string_, str] else round(pred, ndigits=ndigits)}")
        else:
            comparison = "==" if rules['is_categorical'] else "<="
            not_comparison = "!=" if rules['is_categorical'] else ">"
            feature_idx = rules['feature_idx']
            coefficients = rules['coefficients'] if 'coefficients' in rules else None
            thr = rules['threshold']
            
            print(feature_idx)

            if isinstance(feature_idx, (int, np.integer)):
                print(f"{indentation}|--- {rules['textual_rule']}")
                cls.print_rules(rules=rules['left_node'], columns_names=columns_names, indent=indent + 1)

                print(f"{indentation}|--- {rules['not_textual_rule']}")
                cls.print_rules(rules=rules['right_node'], columns_names=columns_names, indent=indent + 1)
                
            elif (isinstance(feature_idx, str) and feature_idx.endswith('_P')): #TODO: cambia
                print(f"{indentation}|--- {names_pivots(feature_idx[:-2])} {comparison} "
                      f"{thr if isinstance(thr, (np.str_, np.string_, str)) else round(thr, ndigits=ndigits)}"
                      f"\t{rules['samples']}")
                cls.print_rules(rules=rules['left_node'], columns_names=columns_names, indent=indent + 1)
            
                print(f"{indentation}|--- {names_pivots(feature_idx[:-2])} {not_comparison} "
                      f"{thr if isinstance(thr, (np.str_, np.string_, str)) else round(thr, ndigits=ndigits)}")
                cls.print_rules(rules=rules['right_node'], columns_names=columns_names, indent=indent + 1)

            
            elif isinstance(feature_idx, tuple) and all(isinstance(i, (int, np.integer)) for i in feature_idx):
                
                print(f"{indentation}|--- near {names_pivots(feature_idx[0])}"
                      
                      f"\t{rules['samples']}")
                cls.print_rules(rules=rules['left_node'], columns_names=columns_names, indent=indent + 1)

                print(f"{indentation}|--- near {names_pivots(feature_idx[1])}"
                     )
                cls.print_rules(rules=rules['right_node'], columns_names=columns_names, indent=indent + 1)
                
            elif isinstance(feature_idx, list) and all(isinstance(i, (int, np.integer)) for i in feature_idx):
                feature_coefficient_pairs = [f"{round(coef, ndigits=ndigits)} * {names(idx)}" for coef, idx in zip(coefficients, feature_idx)]
                feature_coefficient_str = " + ".join(feature_coefficient_pairs)
                
                print(f"{indentation}|--- {feature_coefficient_str} {comparison} "
                      f"{thr if isinstance(thr, (str, np.str_, np.string_)) else round(thr, ndigits=ndigits)}"
                      f"\t{rules['samples']}")
                cls.print_rules(rules=rules['left_node'], columns_names=columns_names, indent=indent + 1)
                
                print(f"{indentation}|--- {feature_coefficient_str} {not_comparison} "
                      f"{thr if isinstance(thr, (str, np.str_, np.string_)) else round(thr, ndigits=ndigits)}")
                cls.print_rules(rules=rules['right_node'], columns_names=columns_names, indent=indent + 1)
                
            elif isinstance(feature_idx, list) and feature_idx[0].endswith('_P'):
                feature_coefficient_pairs = [f"{round(coef, ndigits=ndigits)} * {names_pivots(idx[:-2])}" for coef, idx in zip(coefficients, feature_idx)]
                feature_coefficient_str = " + ".join(feature_coefficient_pairs)
                
                print(f"{indentation}|--- {feature_coefficient_str} {comparison} "
                      f"{thr if isinstance(thr, (str, np.str_, np.string_)) else round(thr, ndigits=ndigits)}"
                      f"\t{rules['samples']}")
                cls.print_rules(rules=rules['left_node'], columns_names=columns_names, indent=indent + 1)
                
                print(f"{indentation}|--- {feature_coefficient_str} {not_comparison} "
                      f"{thr if isinstance(thr, (str, np.str_, np.string_)) else round(thr, ndigits=ndigits)}")
                cls.print_rules(rules=rules['right_node'], columns_names=columns_names, indent=indent + 1)
                
            else: #if list -> oblique
                raise Exception("Unimplemented")

    @classmethod
    def decode_ruletree(cls, vector):

        #need to check if n_classes_ is actually necessary
        
        #n_classes_ = np.array([n_classes_], dtype=np.intp)
        
        idx_to_node = {index: RuleTreeNode(node_id=None,
                                        prediction=None, prediction_probability=None, parent=-1) 
                       for index in range(len(vector[0]))}
        
        idx_to_node[0].node_id = 'R'
        idx_to_node[0].parent = -1
        
        return idx_to_node
                            
    def encode_ruletree(self):
        nodes = (2 ** (self.max_depth + 1)) - 1
        vector = np.zeros((2, nodes), dtype=object)
        
        index = {'R': 0}  # root index
        parent = {}
        
        if not hasattr(self, 'root'):
            raise ValueError('This RuleTree instance must be fitted before encoding.')
        else:
            root_node = self.root
        root_node.encode_node(index, parent, vector, self)
        
        
        for node in range(vector.shape[1]):
            if vector[0][node] == -1:
                # children update
                if ((2*node + 1) < (vector.shape[1] - 1)) and ((2*node + 2) < vector.shape[1]):
                    vector[0][2*node + 1] = -1
                    vector[1][2*node + 1] = vector[1][node]
                    vector[0][2*node + 2] = -1
                    vector[1][2*node + 2] = vector[1][node]

                    parent[2*node + 1] = node
                    parent[2*node + 2] = node

                    # update the node itself
                    vector[0][node] = vector[0][parent[node]]
                    vector[1][node] = vector[1][parent[node]]
                    
        self.vector = vector
        
        return vector


    def to_dict(self, filename=None):
        node_list = [self.root]

        args = {
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_samples_split": self.min_samples_split,
            "max_depth": self.max_depth,
            "prune_useless_leaves": self.prune_useless_leaves,
            "base_stumps": None,
            "stump_selection": self.stump_selection,
            "random_state": self.random_state,
        }

        warnings.warn("As for now base_stump is not serializable")

        dictionary = {
            "tree_type": self.__class__.__module__,
            "args": args,
            "classes_": self.classes_.tolist(),
            "n_classes_": self.n_classes_,
            "nodes": [],
        }

        while len(node_list) > 0:
            node = node_list.pop()
            if not node.is_leaf():
                node_list += [node.node_l, node.node_r]

            dictionary["nodes"].append(node.node_to_dict())

        with open(filename, 'w') as f:
            json.dump(dictionary, f, cls=json_NumpyEncoder)

        return dictionary

    @classmethod
    def from_dict(cls, filename):
        with open(filename, 'r') as f:
            dictionary = json.load(f)

        assert 'tree_type' in dictionary

        class_c = getattr(importlib.import_module(dictionary['tree_type']), dictionary['tree_type'].split('.')[-1])
        tree = class_c(**dictionary.get('args', dict()))
        tree.classes_ = dictionary.get('classes_', np.nan)
        tree.n_classes_ = dictionary.get('n_classes_', np.nan)

        nodes = {node["node_id"]: RuleTreeNode.dict_to_node(node) for node in dictionary['nodes']}

        for node_instance, node_info in zip(nodes.values(), dictionary['nodes']):
            if not node_info["is_leaf"]:
                node_instance.node_l = nodes[node_info["left_node"]]
                node_instance.node_r = nodes[node_info["right_node"]]
                nodes[node_info["left_node"]].parent = node_instance
                nodes[node_info["right_node"]].parent = node_instance

        tree.root = nodes['R']

        return tree

    def export_graphviz(self, columns_names=None, scaler=None, float_precision=3, filename:str|None=None):
        dot = self.root.export_graphviz(columns_names=columns_names, scaler=scaler, float_precision=float_precision)
        if filename is None:
            dot.render(directory=TemporaryDirectory(delete=False).name, view=True)
            return dot
        else:
            dot.render(filename=filename)

