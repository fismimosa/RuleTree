from itertools import product

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

from RuleTree.stumps.classification.PartialPivotTreeStumpClassifier import PartialPivotTreeStumpClassifier
from RuleTree.stumps.classification.PartialProximityTreeStumpClassifier import PartialProximityTreeStumpClassifier

n_jobs=32

max_depth_trees = [
    #1,
    3,
    #5,
    #10,
]

available_stumps = {
    'PartialPivotTreeStumpClassifier': PartialPivotTreeStumpClassifier,
    'PartialProximityTreeStumpClassifier': PartialProximityTreeStumpClassifier,
}

hyper_models = {
    'SPT': {
        'max_depth': max_depth_trees,
        'prune_useless_leaves': [True],
        'stump_selection': ['best',],
        'random_state': [42],
        'splitter': ['best'],
        'base_stumps': [['PartialPivotTreeStumpClassifier']],
        'distance_measure': ['euclidean']
    },
    'PSPT': {
        'max_depth': max_depth_trees,
        'prune_useless_leaves': [True],
        'stump_selection': ['best'],
        'random_state': [42],
        'splitter': ['best'],
        'base_stumps': [['PartialProximityTreeStumpClassifier']],
        'distance_measure': ['euclidean']
    },
}

hyper_stumps = {
    'PartialPivotTreeStumpClassifier': {
        'n_shapelets': [
            10,
            100,
            1000,
            np.inf
        ],
        'n_ts_for_selection': [
            10,
            100,
            1000,
            .1,
            .5,
            np.inf
        ],
        'n_features_strategy': [
            1, 2, 3, 4, 5, # esatti
            #(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), # 1 up to 6 (excluded)
            'elbow', 'drop', 'new', #idea apriori
        ],
        'selection': [
            'all',
            'random',
            'cluster'
        ],
        'scaler': [None, StandardScaler(), MinMaxScaler(), RobustScaler(), MaxAbsScaler()],
        'n_jobs': [n_jobs],
        'random_state': [42],
    },
    'PartialProximityTreeStumpClassifier': {
        'n_shapelets': [
            10,
            100,
            1000,
            np.inf
        ],
        'n_ts_for_selection': [
            10,
            100,
            1000,
            .1,
            .5,
            np.inf
        ],
        'n_features_strategy': [
            1, 2, 3, 4, 5, # esatti
            #(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), # 1 up to 6 (excluded)
            'elbow', 'drop', 'new', #'all',
        ],
        'selection': [
            'all',
            'random',
            'cluster'
        ],
        'proximity_on_same_features': [
            True,
            #False
        ],
        'scaler': [None, StandardScaler(), MinMaxScaler(), RobustScaler(), MaxAbsScaler()],
        'n_jobs': [n_jobs],
        'random_state': [42]
    },
}

def get_hyperparameters(df: pd.DataFrame):
    for model in hyper_models.keys():
        for hyper in product(*hyper_models[model].values()):
            hyper_dict = dict(zip(hyper_models[model].keys(), hyper))
            if model in ['PSPT', 'SPT']:
                if 'PSPT' == model:
                    continue
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

if __name__ == '__main__':
    print(len(list(get_hyperparameters(None))))
