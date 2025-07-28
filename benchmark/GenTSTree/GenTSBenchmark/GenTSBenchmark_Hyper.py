from itertools import product

import keras
import numpy as np
import pandas as pd

n_jobs=32

max_depth_trees = [
    #1,
    4,
    #5,
    #10,
    #None
]

window_sizes = [12, 24, 36, 48, .1, .3, .5, 1.0]
n_estimators = [5, 25, 100, 1000]

shapelet_hyperparameters = { #pyts implementation
    'window_size': window_sizes + ['auto'],
    'criterion': ['mutual_info', 'anova'],
    'window_steps': [1],
    'remove_similar': [True, False],
    'random_state': [42],
    'n_jobs': [n_jobs],
}


hyper_transformations = {
    'shapelet': shapelet_hyperparameters,

}


hyper_models = {
    'DT': {  # the transformation is done at each stump level
        'max_depth': max_depth_trees,
        'prune_useless_leaves': [True],
        'stump_selection': ['best',],
        'random_state': [42],
        'splitter': ['best'],
        'transformation': ['shapelet', 'none'],
    },
    'pre_DT': {  # the transformation step happens before the DT
        'max_depth': max_depth_trees,
        'prune_useless_leaves': [True],
        'stump_selection': ['best',],
        'random_state': [42],
        'splitter': ['best'],
    },
    'GenTS': {  # gentree with shapelets
        'max_depth': max_depth_trees,
        'prune_useless_leaves': [True],
        'stump_selection': ['best',],
        'random_state': [42],
        'splitter': ['best'],
    },
    'Arsenal': {
        'n_kernels': [2000],
        'n_estimators': n_estimators,
        'rocket_transform': ['rocket', 'minirocket', 'multirocket'],
        'max_dilations_per_kernel': [32],
        'n_jobs': [n_jobs],
        'random_state': [42],
    },
    'Hydra': {
        'n_kernels': [8],
        'n_groups': [64],
        'n_jobs': [n_jobs],
        'random_state': [42],
    },
    'MultiRocketHydraClassifier': {
        'n_kernels': [8],
        'n_groups': [64],
        'n_jobs': [n_jobs],
        'random_state': [42],
    },
    'RocketClassifier': {
        'n_kernels': [10000],
        'estimator': [None],
        'n_jobs': [n_jobs],
        'random_state': [42],
    },
    'MiniRocketClassifier': {
        'n_kernels': [10000],
        'max_dilations_per_kernel': [32],
        'estimator': [None],
        'n_jobs': [n_jobs],
        'random_state': [42],
    },
    'MultiRocketClassifier': {
        'n_kernels': [10000],
        'max_dilations_per_kernel': [32],
        'n_features_per_kernel': [4],
        'estimator': [None],
        'n_jobs': [n_jobs],
        'random_state': [42],
    },
    'ResNet': {
        'n_residual_blocks': [3],
        'n_conv_per_residual_block': [3],
        'n_filters': [128, 64, 64],
        'kernel_sizes': [8, 5, 3],
        'strides': [1],
        'dilation_rate': [1],
        'padding': ['padding'],
        'aggregation': ['relu'],
        'use_bias': [True],
        'n_epochs': [1500],
        'batch_size': [16],
        'use_mini_batch_size': [False],
        'callbacks': [keras.callbacks.EarlyStopping(patience=10)],
        'random_state': [42],
        'loss': ['categorical_crossentropy'],
        'metrics': ['accuracy'],
    },
    'MLPClassifier': {
        'n_layers': [3],
        'n_units': [500],
        'activation': ['relu'],
        'use_bias': [True],
        'n_epochs': [2000],
        'batch_size': [16],
        'use_mini_batch_size': [False],
        'callbacks': [keras.callbacks.EarlyStopping(patience=10)],
        'random_state': [42],
        'loss': ['categorical_crossentropy'],
        'metrics': ['accuracy'],
    },
    'InceptionTimeClassifier': {
        'n_classifiers': [5],
        'n_filters': [32],
        'n_conv_per_layer': [3],
        'kernel_size': [40],
        'use_max_pooling': [True],
        'max_pool_size': [3],
        'strides': [1],
        'dilation_rate': [1],
        'padding': ['same'],
        'activation': ['relu'],
        'use_bias': [True],
        'use_residual': [True],
        'use_bottleneck': [True],
        'bottleneck_size': [32],
        'depth': [6],
        'use_custom_filters': [False],
        'n_epochs': [1500],
        'batch_size': [16],
        'use_mini_batch_size': [False],
        'callbacks': [keras.callbacks.EarlyStopping(patience=10)],
        'random_state': [42],
        'loss': ['categorical_crossentropy'],
        'metrics': ['accuracy'],
    },
    'BOSSEnsemble': {
        'threshold': [.92],
        'max_ensemble_size': [500],
        'max_win_len_prop': [1],
        'min_window': [10],
        'feature_selection': ['chi2', 'none', 'random'],
        'use_boss_distance': [True],
        'alphabet_size': [4],
        'n_jobs': [1],
        'random_state': [42],
    },
    'MrSEQLClassifier': {
        'seql_mode': ['fs', 'clf'],
        'symrep': ['sax', 'sfa'],
        'custom_config': [None],
    },
    'MrSQMClassifier': {
        'strat': ['R', 'S', 'RS'],
        'features_per_rep': [500],
        'selection_per_rep': [2000],
        'nsax': [0, 5],
        'nsfa': [0, 5],
        'custom_config': [None],
        'random_state': [42],
        'sfa_norm': [True],
    },
    'MUSE': {
        'anova': [True],
        'variance': [False],
        'bigrams': [True],
        'window_inc': [2],
        'alphabet_size': [4],
        'use_first_order_differences': [True],
        'feature_selection': ['chi2', 'none', 'random'],
        'p_threshold': [.05],
        'support_probabilities': [False],
        'random_state': [42],
        'n_jobs': [n_jobs],
    },
    'WEASEL': {
        'anova': [True],
        'bigrams': [True],
        'binning_strategy': ['information-gain', 'equi-depth', 'equi-width'],
        'window_inc': [2],
        'p_threshold': [.05],
        'alphabet_size': [4],
        'feature_selection': ['chi2', 'none', 'random'],
        'support_probabilities': [False],
        'random_state': [42],
        'n_jobs': [n_jobs],
    },
    'WEASELV2': {
        'min_window': [4],
        'norm_options': [False],
        'word_lengths': [(7, 8)],
        'use_first_differences': [(True, False)],
        'feature_selection': ['chi2_top_k', 'none', 'random'],
        'max_feature_count': [30000],
        'random_state': [42],
        'n_jobs': [n_jobs],
    },
    'KNeighborsTimeSeriesClassifier': {
        'distance': ['dtw'],
        'distance_params': [None],
        'n_neighbors': [1, 3, 10],
        'weights': ['uniform', 'distance'],
        'n_jobs': [n_jobs],
    },
    'ProximityForest': {
        'n_trees': n_estimators,
        'n_splitters': [5],
        'max_depth': max_depth_trees,
        'random_state': [42],
        'n_jobs': [n_jobs],
    },
    'ProximityTree': {
        'n_splitters': [5],
        'max_depth': max_depth_trees,
        'random_state': [42],
        'n_jobs': [n_jobs],
    },
    'Catch22Classifier_RF': {
        'features': ['all'],
        'catch24': [True],
        'outlier_norm': [True],
        'replace_nans': [True],
        'random_state': [42],
        'n_jobs': [n_jobs],
    },
    'Catch22Classifier_DT': {
        'features': ['all'],
        'catch24': [True],
        'outlier_norm': [True],
        'replace_nans': [True],
        'random_state': [42],
        'n_jobs': [n_jobs],
    },
    'HIVECOTEV1': {
        'random_state': [42],
        'n_jobs': [n_jobs],
    },
    'HIVECOTEV2': {
        'random_state': [42],
        'n_jobs': [n_jobs],
    },
    'CanonicalIntervalForestClassifier': {
        'base_estimator': [None],
        'n_estimators': n_estimators,
        'n_intervals': [4, 'sqrt', 'sqrt-div'],
        'min_interval_length': [3],
        'max_interval_length': [.5, np.inf],
        'att_subsample_size': [8],
        'random_state': [42],
        'n_jobs': [n_jobs],
    },
    'DrCIFClassifier': {
        'base_estimator': [None],
        'n_estimators': n_estimators,
        'n_intervals': [4, 'sqrt', 'sqrt-div'],
        'min_interval_length': [3],
        'max_interval_length': [.5, np.inf],
        'att_subsample_size': [10],
        'random_state': [42],
        'n_jobs': [n_jobs],
    },
    'BORF': {
        'window_size_min_window_size': [4],
        'window_size_max_window_size': [None],
        'word_lengths_n_word_lengths': [3],
        'alphabets_min_symbols': [3],
        'alphabets_max_symbols': [4],
        'alphabets_step': [1],
        'dilations_min_dilation': [1],
        'dilations_max_dilation': [None],
        'min_window_to_signal_std_ratio': [.0],
        'n_jobs': [n_jobs],
        'n_jobs_numba': [1],
        'transformer_weights': [None],
        'complexity': ['quadratic', 'linear'],
        'densify': [False],
        'RidgeCV': {
            'alphas': [(0.1, 1.0, 10.0)],
            'fit_intercept': [True],
            'scoring': [None],
            'cv': [None],
            'gcv_mode': ['auto'],
        }
    },
    'IntervalForestClassifier': { #TODO
        'base_estimator': [None],
        'n_estimators': n_estimators,
        'n_intervals': [4, 'sqrt', 'sqrt-div'],
        'min_interval_length': [3],
        'max_interval_length': [.5, np.inf],
        'att_subsample_size': [10],
        'random_state': [42],
        'n_jobs': [n_jobs],
    },
}


def get_hyperparameters(df: pd.DataFrame):
    for model in hyper_models.keys():
        for hyper in product(*hyper_models[model].values()):
            hyper_dict = dict(zip(hyper_models[model].keys(), hyper))
            yield model, hyper_dict, dict()

if __name__ == '__main__':
    print(len(list(get_hyperparameters(None))))
