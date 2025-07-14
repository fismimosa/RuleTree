"""
Module containing the RuleTreeClassifier class.

The RuleTreeClassifier class extends the RuleTree to implement classification-specific
functionality, including methods for handling class probabilities and medoid computation.
"""

import heapq
import traceback
from typing import Union, Optional

import numpy as np
import sklearn
from sklearn import tree
from sklearn.base import ClassifierMixin
import copy

from sklearn.preprocessing import StandardScaler

from RuleTree.exceptions import NoSplitFoundWarning
#from RuleTree.stumps.classification.MultiplePivotTreeStumpClassifier import MultiplePivotTreeStumpClassifier
#from RuleTree.stumps.classification.PivotTreeStumpClassifier import PivotTreeStumpClassifier
#from RuleTree.stumps.classification.ObliquePivotTreeStumpClassifier import ObliquePivotTreeStumpClassifier
from RuleTree.tree.RuleTree import RuleTree
from RuleTree.tree.RuleTreeNode import RuleTreeNode
from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from RuleTree.utils.data_utils import calculate_mode, get_info_gain


class RuleTreeClassifier(RuleTree, ClassifierMixin):
    """
    RuleTree implementation for classification tasks.

    This class extends the RuleTree to include classification-specific methods
    and attributes, such as handling class probabilities and computing medoids.

    Attributes:
        criterion (str): Splitting criterion ('gini' or 'entropy').
        splitter (str): Split strategy ('best' or 'random').
        min_samples_leaf (int): Minimum samples required in a leaf node.
        distance_measure (str): Distance metric for medoid computation.
    """
    def __init__(self,
                 max_leaf_nodes=float('inf'),
                 min_samples_split=2,
                 max_depth=float('inf'),
                 prune_useless_leaves=False,
                 base_stumps: Union[ClassifierMixin, list] = None,
                 stump_selection: str = 'random',
                 random_state=None,
                 distance_measure = None
                 ):
        """
        Initialize the RuleTreeClassifier.

        Args:
            max_leaf_nodes (int): Maximum number of leaf nodes.
            min_samples_split (int): Minimum samples required to split a node.
            max_depth (int): Maximum depth of the tree.
            prune_useless_leaves (bool): Whether to prune useless leaves.
            base_stumps (list): List of base stumps or their configurations.
            stump_selection (str): Stump selection strategy ('random' or 'best').
            random_state (int): Random seed for reproducibility.
            distance_measure (str): Distance metric for medoid computation.
        """
        if base_stumps is None:
            base_stumps = DecisionTreeStumpClassifier(max_depth=1)

        super().__init__(max_leaf_nodes=max_leaf_nodes,
                         min_samples_split=min_samples_split,
                         max_depth=max_depth,
                         prune_useless_leaves=prune_useless_leaves,
                         base_stumps=base_stumps,
                         stump_selection=stump_selection,
                         random_state=random_state,
                         distance_measure=distance_measure)

    def is_split_useless(self, X, clf: tree, idx: np.ndarray):
        """
        Check if a split is useless.

        Args:
            X (np.ndarray): Feature matrix.
            clf (tree): Classifier used for splitting.
            idx (np.ndarray): Indices of the samples to consider.

        Returns:
            bool: True if the split is useless, False otherwise.
        """
        labels = clf.apply(X[idx])

        return len(np.unique(labels)) == 1

    def check_additional_halting_condition(self, y, curr_idx: np.ndarray):
        """
        Check additional halting condition for splitting.

        Args:
            y (np.ndarray): Target labels.
            curr_idx (np.ndarray): Indices of the samples to consider.

        Returns:
            bool: True if halting condition is met, False otherwise.
        """
        return len(np.unique(y[curr_idx])) == 1  # only 1 target

    def queue_push(self, node: RuleTreeNode, idx: np.ndarray):
        """
        Push a node into the priority queue.

        Args:
            node (RuleTreeNode): Node to push.
            idx (np.ndarray): Indices of the samples associated with the node.
        """
        heapq.heappush(self.queue, (len(node.node_id), next(self.tiebreaker), idx, node))

    def make_split(self, X: np.ndarray, y, idx: np.ndarray, medoids_index = None, sample_weight=None, **kwargs) -> tree:
        """
        Create a split using the selected stump.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.
            idx (np.ndarray): Indices of the samples to consider.
            medoids_index (list): Indices of medoids.
            sample_weight (np.ndarray): Sample weights.
            **kwargs: Additional arguments for splitting.

        Returns:
            tree: The selected stump after fitting.
        """
        
    
        if self.stump_selection == 'random':
            stump = self._get_random_stump(X)
        
            stump.fit(X=X,
                      y=y,
                      idx=idx,
                      context = self,
                      sample_weight=None if sample_weight is None else sample_weight[idx])
                
        elif self.stump_selection == 'best':
            clfs = []
            info_gains = []
            for _, stump in self._filter_types(X):
                try:
                    stump = sklearn.clone(stump)

                    stump.fit(X=X,
                              y=y,
                              idx=idx,
                              context=self,
                              sample_weight=None if sample_weight is None else sample_weight[idx])

                    gain = get_info_gain(stump)
                    info_gains.append(gain)

                    clfs.append(stump)
                except ValueError as e:
                    traceback.print_exc()
                    if 'split' in str(e) or 'n_shapelets' in str(e):
                        continue
                    else:
                        raise e
                except NoSplitFoundWarning:
                    # If no split is found, we skip this stump
                    continue
                except Exception as e:
                    traceback.print_exc()
                    continue

            stump = clfs[np.argmax(info_gains)]
        else:
            raise TypeError('Unknown stump selection method')

        return stump

    def prepare_node(self, y: np.ndarray, idx: np.ndarray, node_id: str, node: Optional[RuleTreeNode] = None) -> RuleTreeNode:
        """
        Prepare a node with predictions and probabilities.

        Args:
            y (np.ndarray): Target labels.
            idx (np.ndarray): Indices of the samples to consider.
            node_id (str): Identifier for the node.

        Returns:
            RuleTreeNode: The prepared node.
        """
        prediction = calculate_mode(y[idx])
        predict_proba = np.zeros((len(self.classes_), ))
        for i, classe in enumerate(self.classes_):
            predict_proba[i] = sum(np.where(y[idx] == classe, 1, 0)) / len(y[idx])

        if node is not None:
            node.prediction = prediction
            node.predict_proba = predict_proba
            node.classes = self.classes_
            node.samples = len(y[idx])

            return node


        return RuleTreeNode(
            node_id=node_id,
            prediction=prediction,
            prediction_probability=predict_proba,
            classes = self.classes_,
            n_features=self.n_features,
            parent=None,
            stump=None,
            node_l=None,
            node_r=None,
            samples=len(y[idx]),
        )

    def fit(self, X: np.array, y: np.array = None, sample_weight=None, **kwargs):
        """
        Fit the RuleTreeClassifier to the provided data.

        Args:
            X (np.array): Feature matrix.
            y (np.array): Target labels.
            sample_weight (np.array): Sample weights.
            **kwargs: Additional arguments for fitting.

        Returns:
            RuleTreeClassifier: The fitted RuleTreeClassifier instance.
        """
        super().fit(X, y, sample_weight=sample_weight, **kwargs)

        return self


    def predict_proba(self, X: np.ndarray):
        """
        Predict class probabilities for the input data.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        _, _, proba = self.root.predict(X)

        return proba


    def get_pivots(self, current_node=None, pivot_dicts=None):
        """
        Retrieve pivot information from the tree.

        Args:
            current_node (RuleTreeNode): Current node in the tree.
            pivot_dicts (dict): Dictionary to store pivot information.

        Returns:
            dict: Pivot information for the tree.
        """
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
        """
        Get the base class for stumps.

        Returns:
            type: The base class for stumps.
        """
        return ClassifierMixin

    def _get_prediction_probas(self, current_node = None, probas=None):
        """
        Retrieve prediction probabilities from the tree.

        Args:
            current_node (RuleTreeNode): Current node in the tree.
            probas (list): List to store probabilities.

        Returns:
            list: Prediction probabilities for the tree.
        """
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
        """
        Perform local interpretation of the tree.

        Args:
            X (np.ndarray): Feature matrix.
            joint_contribution (bool): Whether to consider joint contributions.

        Returns:
            np.ndarray: Contributions for each feature and class.
        """
        leaves, paths, leaf_to_path, values = super().local_interpretation(X = X,
                                                                           joint_contribution = joint_contribution)
        normalizer = values.sum(axis=1)[:, np.newaxis]
        normalizer[np.isclose(normalizer, 0.0, rtol=1e-09, atol=1e-09)] = 1.0
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
        """
        Retrieve balanced stumps from the tree.

        Args:
            current_node (RuleTreeNode): Current node in the tree.
            stumps (dict): Dictionary to store balanced stumps.
            p (float): Threshold for balance score.

        Returns:
            dict: Balanced stumps in the tree.
        """
        if stumps is None:
            stumps = {}
        if current_node is None:
            current_node = self.root
            
        if not current_node.is_leaf() and current_node.balance_score > p:
                stumps[current_node.node_id] = (current_node, current_node.balance_score)
                self.get_balanced_stumps(current_node=current_node.node_l, stumps=stumps, p=p)
                self.get_balanced_stumps(current_node=current_node.node_r, stumps=stumps, p=p)
                
        return stumps
    
    def stumps_to_trees(self, balanced_nodes):
        """
        Convert balanced stumps to trees.

        Args:
            balanced_nodes (dict): Balanced stumps.

        Returns:
            dict: Trees created from balanced stumps.
        """
        trees = {}
      
        for k, v in balanced_nodes.items():
            rt = self.__class__()
            rt.classes_ = self.classes_
            
            node = copy.deepcopy(v[0])
            node_l = copy.deepcopy(node.node_l)
            node_r = copy.deepcopy(node.node_r)
                        
            feat = tuple(node.stump.feature_original[0])

            thr = (node.stump.threshold_original[0],)
            
            rt.root = node
            rt.root.node_l = node_l
            rt.root.node_r = node_r
            
            rt.root.node_l.make_leaf()
            rt.root.node_r.make_leaf()
            
            rt.root.node_id, rt.root.node_l.node_id, rt.root.node_r.node_id = 'R', 'Rl', 'Rr'
            

            trees[(feat, thr)] = rt
            
        return trees
            
        
        
    
    

        
    
    @classmethod
    def complete_tree(cls, node, X, y, n_classes_):
        """
        Complete the tree by filling missing attributes.

        Args:
            node (RuleTreeNode): Current node in the tree.
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.
            n_classes_ (int): Number of classes.
        """
        classes_ = list(range(n_classes_))
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