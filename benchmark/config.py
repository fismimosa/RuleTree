import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from RuleTree import RuleTreeClassifier, RuleTreeRegressor, RuleTreeCluster, RuleForestClassifier, RuleForestRegressor
from competitors.kmeanstree import KMeansTree

warnings.filterwarnings("ignore")

DATASET_PATH = '../datasets/'
RESULTS_PATH = '../results/'

TASK_CLF = 'CLF'
TASK_REG = 'REG'
TASK_CLU = 'CLU'
TASK_CLC = 'CLC'
TASK_CLR = 'CLR'

NBR_REPEATED_HOLDOUT = 5

task_folder = {
    TASK_CLF: 'CLF/',
    TASK_REG: 'REG/',
    TASK_CLU: 'CLU/',
    TASK_CLC: 'CLF/',
    TASK_CLR: 'REG/',
}

dataset_target_clf = {
    'adult': 'class',
    'ionosphere': 'class',
    'bank': 'give_credit',
    'auction': 'verification.result',
    'vehicle': 'CLASS',
    'wdbc': 'diagnosis',
    'compas-scores-two-years': 'score_text',
    'german_credit': 'default',
    'iris': 'class',
    'titanic': 'Survived',
    'wine': 'quality',
    'fico': 'RiskPerformance',
    'home': 'in_sf',
    'diabetes': 'Outcome',
}

dataset_target_reg = {
    'abalone': 'Rings',
    'auction': 'verification.time',
    'boston': 'MEDV',
    'carprice': 'price',
    'drinks': 'drinks',
    'insurance': 'charges',
    'intrusion': 'Number of Barriers',
    # 'metamaterial': '', # TODO: 2 target?
    'parkinsons_updrs': 'motor_UPDRS',
    'parkinsons_updrs_total': 'total_UPDRS',
    'students': 'Performance Index',
}

dataset_target_clu = {
    '2d-3c-no123': '2',
    '2d-4c-no9': '2',
    '2d-4c_y': '2',
    '2d-10c': '2',
    '2d-20c-no0': '2',
    '2d-d31': '2',
    'aggregation': '2',
    'cure-t0-2000n-2D': '2',
    'cure-t1-2000n-2D': '2',
    'cure-t2-4k': '2',
    'longsquare': '2',
    's-set1': '2',
    's-set2': '2',
    'tetra': '3',
    'triangle1': '2',
    'triangle2': '2',
    'zelnik5_y': '2',
    'zelnik6_y': '2',
}

datasets_target_clu_sup = set(dataset_target_clu.keys())

dataset_target_clu.update(dataset_target_clf)
dataset_target_clu.update(dataset_target_reg)

dataset_feat_drop_clf = {
    'adult': ['fnlwgt', 'education'],
    'ionosphere': [],
    'bank': [],
    'auction': ['verification.time'],
    'vehicle': [],
    'wdbc': [],
    'compas-scores-two-years': ['Unnamed: 0', 'id', 'compas_screening_date', 'age_cat', 'dob', 'decile_score',
                                'c_jail_in', 'c_jail_out', 'c_offense_date', 'c_arrest_date',
                                'c_days_from_compas', 'c_charge_degree', 'c_charge_desc',
                                'r_charge_degree', 'r_days_from_arrest', 'r_offense_date',
                                'r_charge_desc', 'r_jail_in', 'r_jail_out', 'vr_charge_degree', 'type_of_assessment',
                                'decile_score.1', 'screening_date', 'v_type_of_assessment', 'v_decile_score',
                                'v_score_text', 'v_screening_date', 'in_custody', 'out_custody',
                                'start', 'end', 'event'
                                ],
    'german_credit': ['installment_as_income_perc', 'present_res_since', 'credits_this_bank',
                      'people_under_maintenance'],
    'iris': [],
    'titanic': ['PassengerId', 'Cabin_letter', 'Cabin_n'],
    'wine': [],
    'fico': [],
    'home': [],
    'diabetes': []
}

dataset_feat_drop_reg = {
    'abalone': [],
    'auction': ['verification.result'],
    'boston': [],
    'carprice': ['car_ID', 'CarName'],
    'drinks': ['selector'],
    'insurance': [],
    'intrusion': [],
    'metamaterial': [],
    'parkinsons_updrs': ['total_UPDRS'],
    'parkinsons_updrs_total': ['motor_UPDRS'],
    'students': [],
}

dataset_feat_drop_clu = {
    '2d-3c-no123': [],
    '2d-4c-no9': [],
    '2d-4c_y': [],
    '2d-10c': [],
    '2d-20c-no0': [],
    '2d-d31': [],
    'aggregation': [],
    'cure-t0-2000n-2D': [],
    'cure-t1-2000n-2D': [],
    'cure-t2-4k': [],
    'longsquare': [],
    's-set1': [],
    's-set2': [],
    'tetra': [],
    'triangle1': [],
    'triangle2': [],
    'zelnik5_y': [],
    'zelnik6_y': [],
}

dataset_feat_drop_clu.update(dataset_feat_drop_clf)
dataset_feat_drop_clu.update(dataset_feat_drop_reg)

task_method = {
    TASK_CLF:
        {
            'DT': DecisionTreeClassifier,
            'RT': RuleTreeClassifier,
            'RF': RandomForestClassifier,
            'RTF': RuleForestClassifier,
        },  # 'KNN': KNeighborsClassifier(),

    TASK_REG:
        {
            'DT': DecisionTreeRegressor,
            'RT': RuleTreeRegressor,
            'RF': RandomForestRegressor,
            'RTF': RuleForestRegressor,
        },  # 'KNN': KNeighborsRegressor(),

    TASK_CLU:
        {
            'KM': KMeans,
            'RT': RuleTreeCluster,
            'KT': KMeansTree,
        },

    TASK_CLC:
        {
            'KT': KMeansTree,
            'RT': RuleTreeCluster
        },

    TASK_CLR:
        {
            'KT': KMeansTree,
            'RT': RuleTreeCluster
        },
}

N_JOBS = [8]

MAX_DEPTH_LIST = [2, 3, 4, 5, 6, None]
MIN_SAMPLE_SPLIT_LIST = [2, 5, 10, 20, 0.01, 0.05, 0.1]
MIN_SAMPLE_LEAF_LIST = [1, 3, 5, 10, 0.01, 0.05, 0.1]
MAX_LEAF_NODES = [None, 32]
CCP_ALPHA_LIST = [0.0, 0.001, 0.01, 0.1]
RANDOM_STATE_LIST = np.arange(0, 10).tolist()
N_CLUSTERS_LIST = [2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64]
BIC_EPS_LIST = [0.0, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0]
N_ESTIMATORS = [10, 100, 500, 1000]

preprocessing_params = {
    'one_hot_encode_cat': [True, False],
    'max_n_vals': [np.inf],
    'max_n_vals_cat': [0, 20]
}

methods_params_clf = {
    'DT': {
        'criterion': ['gini'],
        'splitter': ['best'],
        'max_depth': MAX_DEPTH_LIST,
        'min_samples_split': MIN_SAMPLE_SPLIT_LIST,
        'min_samples_leaf': MIN_SAMPLE_LEAF_LIST,
        'min_weight_fraction_leaf': [0.0],
        'max_features': [None],
        'max_leaf_nodes': MAX_LEAF_NODES,
        'min_impurity_decrease': [0.0],
        'class_weight': [None],
        'ccp_alpha': CCP_ALPHA_LIST,
        'random_state': RANDOM_STATE_LIST,
    },
    'RT': {
        'max_leaf_nodes': MAX_LEAF_NODES,
        'min_samples_split': MIN_SAMPLE_SPLIT_LIST,
        'max_depth': MAX_DEPTH_LIST,
        'prune_useless_leaves': [False, True],
        'random_state': RANDOM_STATE_LIST,
        'criterion': ['gini'],
        'splitter': ['best'],
        'min_samples_leaf': MIN_SAMPLE_LEAF_LIST,
        'min_weight_fraction_leaf': [0.0],
        'max_features': [None],
        'min_impurity_decrease': [0.0],
        'class_weight': [None],
        'ccp_alpha': CCP_ALPHA_LIST,
    },

    'RF': {
        'n_estimators': N_ESTIMATORS,
        'criterion': ['gini'],
        'max_depth': MAX_DEPTH_LIST,
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'min_weight_fraction_leaf': [0.0],
        'min_impurity_decrease': [0.0],
        'max_leaf_nodes': MAX_LEAF_NODES,
        'class_weight': [None],
        'ccp_alpha': CCP_ALPHA_LIST,
        'max_samples': [None],
        'max_features': [None],
        'bootstrap': [True],
        'oob_score': [False],
        'warm_start': [False],
        'n_jobs': N_JOBS,
        'random_state': RANDOM_STATE_LIST,
    },

    'RTF': {
        'n_estimators': N_ESTIMATORS,
        'criterion': ['gini'],
        'splitter': ['best'],
        'max_depth': MAX_DEPTH_LIST,
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'min_weight_fraction_leaf': [0.0],
        'min_impurity_decrease': [0.0],
        'max_leaf_nodes': MAX_LEAF_NODES,
        'class_weight': [None],
        'ccp_alpha': CCP_ALPHA_LIST,
        'prune_useless_leaves': [False, True],
        'max_samples': [None],
        'max_features': [None],
        'bootstrap': [True],
        'oob_score': [False],
        'warm_start': [False],
        'n_jobs': N_JOBS,
        'random_state': RANDOM_STATE_LIST,
    }
}

methods_params_reg = {
    'DT': {
        'criterion': ['squared_error'],
        'splitter': ['best'],
        'max_depth': MAX_DEPTH_LIST,
        'min_samples_split': MIN_SAMPLE_SPLIT_LIST,
        'min_samples_leaf': MIN_SAMPLE_LEAF_LIST,
        'min_weight_fraction_leaf': [0.0],
        'max_features': [None],
        'max_leaf_nodes': MAX_LEAF_NODES,
        'min_impurity_decrease': [0.0],
        'ccp_alpha': CCP_ALPHA_LIST,
        'random_state': RANDOM_STATE_LIST,
    },
    'RT': {
        'max_leaf_nodes': MAX_LEAF_NODES,
        'min_samples_split': MIN_SAMPLE_SPLIT_LIST,
        'max_depth': MAX_DEPTH_LIST,
        'prune_useless_leaves': [False, True],
        'random_state': RANDOM_STATE_LIST,
        'criterion': ['squared_error'],
        'splitter': ['best'],
        'min_samples_leaf': MIN_SAMPLE_LEAF_LIST,
        'min_weight_fraction_leaf': [0.0],
        'max_features': [None],
        'min_impurity_decrease': [0.0],
        'ccp_alpha': CCP_ALPHA_LIST,
    },

    'RF': {
        'n_estimators': N_ESTIMATORS,
        'criterion': ['squared_error'],
        'max_depth': MAX_DEPTH_LIST,
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'min_weight_fraction_leaf': [0.0],
        'min_impurity_decrease': [0.0],
        'max_leaf_nodes': MAX_LEAF_NODES,
        'ccp_alpha': CCP_ALPHA_LIST,
        'max_samples': [None],
        'max_features': [None],
        'bootstrap': [True],
        'oob_score': [False],
        'warm_start': [False],
        'n_jobs': N_JOBS,
        'random_state': RANDOM_STATE_LIST,
    },

    'RTF': {
        'n_estimators': N_ESTIMATORS,
        'criterion': ['squared_error'],
        'splitter': ['best'],
        'max_depth': MAX_DEPTH_LIST,
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'min_weight_fraction_leaf': [0.0],
        'min_impurity_decrease': [0.0],
        'max_leaf_nodes': MAX_LEAF_NODES,
        'ccp_alpha': CCP_ALPHA_LIST,
        'prune_useless_leaves': [False, True],
        'max_samples': [None],
        'max_features': [None],
        'bootstrap': [True],
        'oob_score': [False],
        'warm_start': [False],
        'n_jobs': N_JOBS,
        'random_state': RANDOM_STATE_LIST,
    }
}

methods_params_unsup = {
    'KM': {
        'n_clusters': N_CLUSTERS_LIST,
        'init': ['k-means++'],
        'n_init': [10],
        'max_iter': [300],
        'tol': [0.0001],
        'copy_x': [True],
        'algorithm': ['lloyd'],
        'random_state': RANDOM_STATE_LIST,
    },
    'KT': {
        'labels_as_tree_leaves': [False, True],
        'n_clusters': N_CLUSTERS_LIST,
        'min_samples_split': MIN_SAMPLE_SPLIT_LIST,
        'min_samples_leaf': MIN_SAMPLE_LEAF_LIST,
        'clf_impurity': ['gini'],
        'init': ['k-means++'],
        'n_init': [10],
        'max_iter': [300],
        'tol': [0.0001],
        'copy_x': [True],
        'max_depth': MAX_DEPTH_LIST,
        'algorithm': ['lloyd'],
        'splitter': ['best'],
        'min_weight_fraction_leaf': [0.0],
        'max_features': [None],
        'max_leaf_nodes': MAX_LEAF_NODES,
        'min_impurity_decrease': [0.0],
        'class_weight': [None],
        'ccp_alpha': CCP_ALPHA_LIST,
        'random_state': RANDOM_STATE_LIST,
    },
    'RT': {
        'n_components': [1, 2],
        'clus_impurity': ['r2', 'bic'],
        'bic_eps': BIC_EPS_LIST,
        'max_leaf_nodes': MAX_LEAF_NODES,
        'min_samples_split': MIN_SAMPLE_SPLIT_LIST,
        'min_samples_leaf': MIN_SAMPLE_LEAF_LIST,
        'max_depth': MAX_DEPTH_LIST,
        'prune_useless_leaves': [False, True],
        'min_weight_fraction_leaf': [0.0],
        'min_impurity_decrease': [0.0],
        'ccp_alpha': CCP_ALPHA_LIST,
        'max_features': [None],
        'random_state': RANDOM_STATE_LIST,
    }
}
