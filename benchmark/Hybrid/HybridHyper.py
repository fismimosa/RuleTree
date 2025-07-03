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
from benchmark.Hybrid.HybridReaders import read_titanic

max_depth_trees = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    None
]
n_jobs = 16

available_stumps = {
    'DecisionTreeStumpClassifier': DecisionTreeStumpClassifier,
    'ObliqueDecisionTreeStumpClassifier': ObliqueDecisionTreeStumpClassifier,
    'PivotTreeStumpClassifier': PivotTreeStumpClassifier,
    'MultiplePivotTreeStumpClassifier': MultiplePivotTreeStumpClassifier,
    'ObliquePivotTreeStumpClassifier': ObliquePivotTreeStumpClassifier,
    'MultipleObliquePivotTreeStumpClassifier': MultipleObliquePivotTreeStumpClassifier,
    'PartialPivotTreeStumpClassifier': PartialPivotTreeStumpClassifier,
    'PartialProximityTreeStumpClassifier': PartialProximityTreeStumpClassifier,
}

hyper_models = {
    'RT': {
        'max_depth': max_depth_trees,
        'prune_useless_leaves': [True],
        'stump_selection': [
            'best',
            'random'
        ],
        'random_state': [42],
        'splitter': ['best', 'random'],
        'base_stumps': [[x] for x in available_stumps.keys()] + [list(available_stumps.keys())] + [list(available_stumps.keys())[:-2]],
        'distance_measure': ['euclidean']
    },
    'RF': {
        'n_estimators': [10, 100, 1000],
        'random_state': [42],
        'max_depth': max_depth_trees,
        'n_jobs': [n_jobs],
    },
    'AB': {
        'n_estimators': [10, 100, 1000],
        'random_state': [42],
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
            10,
            100,
            np.inf
        ],
        'n_features_strategy': [
            #1,
            2,
            #3,
            'n-1',
            'n-2',
            #'all'
            'sqrt',
            #'half',
            #'auto',
            'gap',
        ],
        'n_jobs': [n_jobs],
        'random_state': [42],
        'selection': [
            #'mi_clf',
            'random',
            'cluster'
        ],
        'n_shapelets_for_selection': [
            .1,
            .5,
            #np.inf
        ],
        'n_ts_for_selection': [
            .1,
            .5,
            #np.inf
        ],
    },
    'PartialProximityTreeStumpClassifier': {
        'n_shapelets': [
            10,
            100,
            np.inf
        ],
        'n_features_strategy': [
            #1,
            2,
            #3,
            'n-1',
            'n-2',
            #'all'
            'sqrt',
            #'half',
            #'auto',
            'gap',
        ],
        'n_jobs': [n_jobs],
        'random_state': [42],
        'selection': [
            #'mi_clf',
            'random',
            'cluster'
        ],
        'n_shapelets_for_selection': [
            .1,
            .5,
            #np.inf
        ],
        'n_ts_for_selection': [
            .1,
            .5,
            #np.inf
        ],
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
    n_features = df.shape[1] -1

    for model in hyper_models.keys():
        for hyper in product(*hyper_models[model].values()):
            hyper_dict = dict(zip(hyper_models[model].keys(), hyper))
            if model == 'RT':
                base_stumps = hyper_dict['base_stumps']
                stump_hyper_lists = [
                    list(product(*hyper_stumps[stump].values()))
                    for stump in base_stumps
                ]
                for combo in product(*stump_hyper_lists):
                    all_stump_hyper = dict()
                    for stump, hyper_stump in zip(base_stumps, combo):
                        hyper_stump_dict = dict(zip(hyper_stumps[stump].keys(), hyper_stump))
                        if 'n_features_strategy' in hyper_stump_dict:
                            if hyper_stump_dict['n_features_strategy'] == 'half':
                                hyper_stump_dict['n_features_strategy'] = n_features // 2
                            elif hyper_stump_dict['n_features_strategy'] == 'n-1':
                                hyper_stump_dict['n_features_strategy'] = n_features - 1
                            elif hyper_stump_dict['n_features_strategy'] == 'n-2':
                                hyper_stump_dict['n_features_strategy'] = n_features - 2

                        all_stump_hyper[stump] = hyper_stump_dict
                    yield model, hyper_dict, all_stump_hyper
            else:
                yield model, hyper_dict, dict()


if __name__ == "__main__":
    _, df = read_titanic()
    print(len(list(get_hyperparameters(df))))
