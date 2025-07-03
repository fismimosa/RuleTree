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
    3,
    #5,
    #10,
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

for stump_to_exclude in available_stumps.keys():
    available_stumps_copy = available_stumps.copy()
    del available_stumps_copy[stump_to_exclude]
    model_name = f"HDT_no_{stump_to_exclude}"

    hyper_models[model_name] = {
        'max_depth': max_depth_trees,
        'prune_useless_leaves': [True],
        'stump_selection': ['best'],
        'random_state': [42],
        'splitter': ['best'],
        'base_stumps': [[x for x in available_stumps_copy.keys() if x != stump_to_exclude]],
        'distance_measure': ['euclidean']
    }

for stump_name in available_stumps.keys():
    hyper_models['ONLY_'+stump_name.replace('TreeStumpClassifier', '')] = {
        'max_depth': max_depth_trees,
        'prune_useless_leaves': [True],
        'stump_selection': ['best'],
        'random_state': [42],
        'splitter': ['best'],
        'base_stumps': [[stump_name]],
        'distance_measure': ['euclidean']
    }

hyper_models |= {
    'HDT': {
        'max_depth': max_depth_trees,
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

    'HDT_no_classic': {
        'max_depth': max_depth_trees,
        'prune_useless_leaves': [True],
        'stump_selection': [
            'best',
            #'random'
        ],
        'random_state': [42],
        'splitter': ['best'],
        'base_stumps': [list(available_stumps.keys())[2:]],
        'distance_measure': ['euclidean']
    },

    'HDT_no_pivot': {
        'max_depth': max_depth_trees,
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

    'HDT_no_partial': {
        'max_depth': max_depth_trees,
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
            'random',
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
            'cluster'
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


def get_hyperparameters(df: pd.DataFrame):
    for model in hyper_models.keys():
        for hyper in product(*hyper_models[model].values()):
            hyper_dict = dict(zip(hyper_models[model].keys(), hyper))
            base_stumps = hyper_dict['base_stumps']
            stump_hyper_lists = [
                list(product(*hyper_stumps[stump].values()))
                for stump in base_stumps
            ]
            for combo in product(*stump_hyper_lists):
                all_stump_hyper = dict()
                for stump, hyper_stump in zip(base_stumps, combo):
                    hyper_stump_dict = dict(zip(hyper_stumps[stump].keys(), hyper_stump))
                    all_stump_hyper[stump] = hyper_stump_dict
                yield model, hyper_dict, all_stump_hyper


if __name__ == "__main__":
    print(len(list(get_hyperparameters(None))))
