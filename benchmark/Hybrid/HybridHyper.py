from itertools import product

import numpy as np
import pandas as pd

from RuleTree.stumps.classification import DecisionTreeStumpClassifier, ObliqueDecisionTreeStumpClassifier, \
    PivotTreeStumpClassifier, MultiplePivotTreeStumpClassifier, ObliquePivotTreeStumpClassifier
from RuleTree.stumps.classification.MultipleObliquePivotTreeStumpClassifier import \
    MultipleObliquePivotTreeStumpClassifier
from RuleTree.stumps.classification.PartialPivotTreeStumpClassifier import PartialPivotTreeStumpClassifier
from RuleTree.stumps.classification.PartialProximityTreeStumpClassifier import PartialProximityTreeStumpClassifier
from benchmark.Hybrid.HybridReaders import read_titanic

max_depth_trees = [2, 3, 10, None]
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
        'stump_selection': ['best', 'random'],
        'random_state': [42],
        'splitter': ['best'],
        'base_stumps': [[x] for x in available_stumps.keys()] + [list(available_stumps.keys())],
        'distance_measure': ['euclidean']
    },
    'DT': {
        'max_depth': max_depth_trees,
        'random_state': [42],
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
        'max_oblique_features': [2, 3],
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
    'ObliquePivotTreeStumpClassifier': {
        'max_depth': [1],
        'random_state': [42],
        'oblique_split_type': ['householder'],
        'pca': [None],
        'max_oblique_features': [2, 3],
        'tau': [1e-4],
    },
    'MultipleObliquePivotTreeStumpClassifier': {
        'max_depth': [1],
        'random_state': [42],
        'oblique_split_type': ['householder'],
        'pca': [None],
        'max_oblique_features': [2, 3],
        'tau': [1e-4],
    },
    'PartialPivotTreeStumpClassifier': {
        'n_shapelets': [5, 10, 100, np.inf],
        'max_n_features': ['all', 'sqrt', 2, 3, 'half', 'n-1'],
        'n_jobs': [n_jobs],
        'random_state': [42],
        'selection': ['mi_clf', 'random', 'cluster'],
    },
    'PartialProximityTreeStumpClassifier': {
        'n_shapelets': [5, 10, 100, np.inf],
        'max_n_features': ['all', 'sqrt', 2, 3, 5, 'half', 'n-1'],
        'n_jobs': [n_jobs],
        'random_state': [42],
        'selection': ['mi_clf', 'random', 'cluster'],
    },
}


def get_hyperparameters(df: pd.DataFrame):
    n_features = df.shape[1]

    for model in hyper_models.keys():
        for hyper in product(*hyper_models[model].values()):
            hyper_dict = dict(zip(hyper_models[model].keys(), hyper))
            if model == 'RT':
                all_stump_hyper = dict()
                for stump in hyper_dict['base_stumps']:
                    for hyper_stump in product(*hyper_stumps[stump].values()):
                        hyper_stump_dict = dict(zip(hyper_stumps[stump].keys(), hyper_stump))
                        if 'max_n_features' in hyper_stump:
                            if hyper_stump_dict['max_n_features'] == 'half':
                                hyper_stump_dict['max_n_features'] = n_features // 2
                            elif hyper_stump_dict['max_n_features'] == 'n-1':
                                hyper_stump_dict['max_n_features'] = n_features - 1
                        all_stump_hyper[stump] = hyper_stump_dict

                yield model, hyper_dict, all_stump_hyper


if __name__ == "__main__":
    _, df = read_titanic()
    for model, hyper_dict, hyper_stump_dict in get_hyperparameters(df):
        print(model, hyper_dict, hyper_stump_dict)
