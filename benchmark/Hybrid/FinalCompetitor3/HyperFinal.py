from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm

from RuleTree.stumps.classification import DecisionTreeStumpClassifier, ObliqueDecisionTreeStumpClassifier, \
    PivotTreeStumpClassifier, MultiplePivotTreeStumpClassifier, ObliquePivotTreeStumpClassifier
from RuleTree.stumps.classification.MultipleObliquePivotTreeStumpClassifier import \
    MultipleObliquePivotTreeStumpClassifier
from RuleTree.stumps.classification.PartialPivotTreeStumpClassifier import PartialPivotTreeStumpClassifier
from RuleTree.stumps.classification.PartialProximityTreeStumpClassifier import PartialProximityTreeStumpClassifier

max_depth_trees = [
    #1,
    #2,
    #3,
    4,
    #5,
    #6,
    #7,
    #8,
    #9,
    #10,
    #None
]

n_jobs = 32

available_stumps = {
    'DecisionTreeStumpClassifier': DecisionTreeStumpClassifier,
    'ObliqueDecisionTreeStumpClassifier': ObliqueDecisionTreeStumpClassifier,
    'PivotTreeStumpClassifier': PivotTreeStumpClassifier,
    'MultiplePivotTreeStumpClassifier': MultiplePivotTreeStumpClassifier,
    'PartialPivotTreeStumpClassifier': PartialPivotTreeStumpClassifier,
    'PartialProximityTreeStumpClassifier': PartialProximityTreeStumpClassifier,
}

hyper_models = dict()


"""
'RFHDT': {
        'n_estimators': [100, 1000],
        'criterion': ['gini'],
        'n_jobs': [n_jobs],
        'max_depth': [None],
        'prune_useless_leaves': [True],
        'stump_selection': [
            'best',
            #'random'
        ],
        'random_state': [42],
        'splitter': ['best'],
        'base_stumps': [list(available_stumps.keys())],
        'distance_measure': ['euclidean']
    },
    'HRFDT': {
        'n_estimators': [100, 1000],
        'criterion': ['gini'],
        'n_jobs': [n_jobs],
        'max_depth': [None],
        'prune_useless_leaves': [True],
        'stump_selection': [
            'best',
            #'random'
        ],
        'random_state': [42],
        'splitter': ['best'],
        'base_stumps': [list(available_stumps.keys())],
        'distance_measure': ['euclidean']
    },

"""

hyper_models |= {

    'HDT': {
        'max_depth': [3], #max_depth_trees,
        'prune_useless_leaves': [True],
        'stump_selection': [
            'best',
            #'random'
        ],
        'random_state': [42],
        'splitter': ['best'],
        'base_stumps': [list(available_stumps.keys())],
        'distance_measure': ['euclidean']
    },
    'HDT_no_partial': {
        'max_depth': [2, 5], #max_depth_trees,
        'prune_useless_leaves': [True],
        'stump_selection': [
            'best',
            #'random'
        ],
        'random_state': [42],
        'splitter': ['best'],
        'base_stumps': [list(available_stumps.keys())[:-2]],
        'distance_measure': ['euclidean']
    },
    'HDT_no_pivot': {
        'max_depth': [1], #max_depth_trees,
        'prune_useless_leaves': [True],
        'stump_selection': [
            'best',
            #'random'
        ],
        'random_state': [42],
        'splitter': ['best'],
        'base_stumps': [list(available_stumps.keys())[:2]+list(available_stumps.keys())[-2:]],
        'distance_measure': ['euclidean']
    },
    'DT': {
        'max_depth': [6],#max_depth_trees,
        'prune_useless_leaves': [True],
        'stump_selection': [
            'best',
            #'random'
        ],
        'random_state': [42],
        'splitter': ['best'],
        'base_stumps': [[DecisionTreeStumpClassifier(max_depth=1)]],
        'distance_measure': ['euclidean']
    },
    'RF': {
        'n_estimators': [100, 1000],
        'criterion': ['gini'],
        'max_depth': [None],
        'n_jobs': [n_jobs]
    },
    'AB': {
        'n_estimators': [50, 100, 1000],
        'learning_rate': [1.]
    },
    'ETC': {
        'splitter': ['random'],
        'criterion': ['gini'],
        'max_depth': [None],
    },
    'ETsC': {
        'n_estimators': [100, 1000],
        'criterion': ['gini'],
        'max_depth': [None],
        'n_jobs': [n_jobs],
    },
    'GBC': {
        'loss': ['log_loss'],
        'learning_rate': [.1],
        'n_estimators': [100, 1000],
        'criterion': ['friedman_mse'],
        'max_depth': [3],
    },
    'HGBC': {
        'loss': ['log_loss'],
        'learning_rate': [.1],
        'max_depth': [3],
        'max_iter': [100]
    },
    # Non sklearn
    'XGB': { #XGBoost
        'booster': ['gbtree', 'gblinear'],
        'n_estimators': [100, 1000],
        'learning_rate': [.3],
        'min_split_loss': [0],
        'max_depth': [6],
        'n_jobs': [n_jobs],
        'verbosity': [0]
    },
    'Dart': { #XGBoost
        'booster': ['dart'],
        'n_estimators': [100, 1000],
        'learning_rate': [.3],
        'min_split_loss': [0],
        'max_depth': [6],
        'n_jobs': [n_jobs],
        'verbosity': [0]
    },
    'LGBM': { #LightGBM
        'boosting_type': ['gbdt'],
        'num_leaves': [31],
        'max_depth': [-1],
        'learning_rate': [.1],
        'n_estimators': [100, 1000],
        'n_jobs': [n_jobs],
    },
    'EBM': { #explainable boosting machine
        'n_jobs': [n_jobs],
    }
}

hyper_stumps = {
    'DecisionTreeStumpClassifier': {
        'max_depth': [1],
        'random_state': [42],
    },
    'ObliqueDecisionTreeStumpClassifier': {
        'max_depth': [1],
        'random_state': [42],
        'oblique_split_type': ['householder'],
        'pca': [None],
        'max_oblique_features': [2],
        'tau': [1e-4],
    },
    'PivotTreeStumpClassifier': {
        'max_depth': [1],
        'random_state': [42],
    },
    'MultiplePivotTreeStumpClassifier': {
        'max_depth': [1],
        'random_state': [42],
    },
    'PartialPivotTreeStumpClassifier': {
        'n_shapelets': [
            #10,
            100,
            #np.inf
        ],
        'n_ts_for_selection': [
            .1,
            #.5,
            #np.inf
        ],
        'n_features_strategy': [
            #1, 2, 3, 4, 5, # esatti
            #(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), # 1 up to 6 (excluded)
            #'elbow', 'drop', 'new', #idea apriori
            'elbow'
        ],
        'selection': [
            #'all',
            #'random',
            'cluster'
        ],
        'n_jobs': [n_jobs],
        'random_state': [42],
    },
    'PartialProximityTreeStumpClassifier': {
        'n_shapelets': [
            #10,
            100,
            #np.inf
        ],
        'n_ts_for_selection': [
            .1,
            #.5,
            #np.inf
        ],
        'n_features_strategy': [
            #1, 2, 3, 4, 5, # esatti
            #(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), # 1 up to 6 (excluded)
            #'elbow', 'drop', 'new', #'all',
            'elbow'
        ],
        'selection': [
            #'all',
            'random',
            #'cluster'
        ],
        'proximity_on_same_features': [True], #False
        'n_jobs': [n_jobs],
        'random_state': [42]
    },
    'ObliquePivotTreeStumpClassifier': {
        'max_depth': [1],
        'random_state': [42],
        'oblique_split_type': ['householder'],
        'pca': [None],
        'max_oblique_features': [2],
        'tau': [1e-4],
    },
    'MultipleObliquePivotTreeStumpClassifier': {
        'max_depth': [1],
        'random_state': [42],
        'oblique_split_type': ['householder'],
        'pca': [None],
        'max_oblique_features': [2],
        'tau': [1e-4],
    },
}


for stump_name in available_stumps.keys():
    max_depth_trees_ = max_depth_trees

    if 'Decision' in stump_name:
        max_depth_trees_ = [6]
    if 'PartialProximity' in stump_name:
        max_depth_trees_ = [4, 7]
    if 'Pivot' in stump_name:
        max_depth_trees_ = [6]
    if 'ObliqueDecision' in stump_name:
        max_depth_trees_ = [6]

    hyper_models['ONLY_'+stump_name.replace('TreeStumpClassifier', '')] = {
        'max_depth': max_depth_trees_,
        'prune_useless_leaves': [True],
        'stump_selection': ['best'],
        'random_state': [42],
        'splitter': ['best'],
        'base_stumps': [[stump_name]],
        'distance_measure': ['euclidean']
    }


def get_hyperparameters(df: pd.DataFrame):
    for model in hyper_models.keys():
        for hyper in product(*hyper_models[model].values()):
            hyper_dict = dict(zip(hyper_models[model].keys(), hyper)) #HDT RFHDT HRFDT
            if model in ['HDT', 'RFHDT', 'HRFDT', 'HDT_no_partial', 'HDT_no_pivot'] or 'ONLY_' in model:
                base_stumps = hyper_dict['base_stumps']
                stump_hyper_lists = [
                    list(product(*hyper_stumps[stump].values()))
                    for stump in base_stumps
                ]

                for combo in product(*stump_hyper_lists):
                    all_stump_hyper = dict()
                    to_skip = False
                    for stump, hyper_stump in zip(base_stumps, combo):
                        hyper_stump_dict = dict(zip(hyper_stumps[stump].keys(), hyper_stump))
                        all_stump_hyper[stump] = hyper_stump_dict

                        if 'n_features_strategy' in hyper_stump_dict:
                            n_feature = hyper_stump_dict['n_features_strategy']
                            if type(n_feature) == int and n_feature >= df.shape[1]-1:
                                to_skip = True
                                break

                            if type(n_feature) == tuple and n_feature[-1] >= df.shape[1]-1:
                                to_skip = True
                                break
                    if not to_skip:
                        yield model, hyper_dict, all_stump_hyper
            else:
                yield model, hyper_dict, dict()


if __name__ == "__main__":
    print(len(list(get_hyperparameters(None))))
