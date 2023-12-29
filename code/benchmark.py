import os
import sys
import time
import hashlib
import datetime
import itertools

import numpy as np
import pandas as pd

import sklearn.metrics as skm

from ruletree import RuleTree
from ruletree import prepare_data
from kmeanstree import KMeansTree

from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset_path = '../datasets/'
results_path = '../results/'

TASK_CLF = 'CLF'
TASK_REG = 'REG'
TASK_CLU = 'CLU'
TASK_CLC = 'CLC'
TASK_CLR = 'CLR'

TASK_TYPE = TASK_CLF
NBR_REPEATED_HOLDOUT = 5
DATASETS_STATS_FILENAME = '%s_datasets_stats.csv' % TASK_TYPE
RESULT_FILENAME = '%s_results.csv' % TASK_TYPE

task_folder = {
    TASK_CLF: 'classification/',
    TASK_REG: 'regression/',
    TASK_CLU: 'clustering/',
    TASK_CLC: 'classification/',
    TASK_CLR: 'regression/',
}

dataset_target = {
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

dataset_feat_drop = {
    'adult': ['fnlwgt', 'education'],
    'ionosphere': [],
    'bank': [],
    'auction': ['verification.time'],
    'vehicle': [],
    'wdbc': [],
    'compas-scores-two-years': ['Unnamed: 0',  'id', 'compas_screening_date', 'age_cat', 'decile_score',
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

task_method = {
    TASK_CLF: {'DT': DecisionTreeClassifier(), 'RT': RuleTree()},  # 'KNN': KNeighborsClassifier(),
    TASK_REG: {'DT': DecisionTreeRegressor(), 'RT': RuleTree()},   # 'KNN': KNeighborsRegressor(),
    TASK_CLU: {'KM': KMeans(), 'KT': KMeansTree(), 'RT': RuleTree()},
    TASK_CLC: {'KT': KMeansTree(), 'RT': RuleTree()},
    TASK_CLR: {'KT': KMeansTree(), 'RT': RuleTree()},
}


MAX_DEPTH_LIST = [2, 3, 4, 5, 6, None]
MIN_SAMPLE_SPLIT_LIST = [2, 5, 10, 20, 0.01, 0.05, 0.1]
MIN_SAMPLE_LEAF_LIST = [1, 3, 5, 10, 0.01, 0.05, 0.1]
MAX_LEAF_NODES = [None, 32]
CCP_ALPHA_LIST = [0.0, 0.001, 0.01, 0.1]
RANDOM_STATE_LIST = np.arange(0, 10).tolist()
N_CLUSTERS_LIST = [2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64]
BIC_EPS_LIST = [0.0, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

MAX_NBR_VALUES_CAT = 20  # con meno-uguale di max_nbr_values_cat valori e' categorica
MAX_NBR_VALUES = np.inf
ONE_HOT_ENCODE_CAT = True

methods_params_sup = {
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
        'max_depth': MAX_DEPTH_LIST,
        'max_nbr_nodes': MAX_LEAF_NODES,
        'min_samples_split': MIN_SAMPLE_SPLIT_LIST,
        'min_samples_leaf': MIN_SAMPLE_LEAF_LIST,
        # 'max_nbr_values': [np.inf],
        # 'max_nbr_values_cat': [20],
        'allow_oblique_splits': [False, True],
        'force_oblique_splits': [False, True],
        'max_oblique_features': [2],
        'prune_useless_leaves': [False, True],
        'n_components': [2],
        'bic_eps': [0.0],
        'clus_impurity': ['bic'],
        'clf_impurity': ['gini'],
        'reg_impurity': ['squared_error'],
        'feature_names': [None],
        'exclude_split_feature_from_reduction': [False],
        'precision': [2],
        'cat_precision': [2],
        'n_jobs': [1],
        'random_state': RANDOM_STATE_LIST,
    }
}

methods_params_unsup = {
    'KT': {
        'labels_as_tree_leaves': [False, True],
        # 'max_nbr_values_cat': [20],
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
        'algorithm': ['auto'],
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
        'max_depth': MAX_DEPTH_LIST,
        'max_nbr_nodes': MAX_LEAF_NODES,
        'min_samples_split': MIN_SAMPLE_SPLIT_LIST,
        'min_samples_leaf': MIN_SAMPLE_LEAF_LIST,
        # 'max_nbr_values': [np.inf],
        # 'max_nbr_values_cat': [20],
        'allow_oblique_splits': [False, True],
        'force_oblique_splits': [False, True],
        'max_oblique_features': [2],
        'prune_useless_leaves': [False, True],
        'n_components': [2],
        'bic_eps': BIC_EPS_LIST,
        'clus_impurity': ['r2', 'bic'],
        'clf_impurity': ['gini'],
        'reg_impurity': ['squared_error'],
        'feature_names': [None],
        'exclude_split_feature_from_reduction': [False, True],
        'precision': [2],
        'cat_precision': [2],
        'n_jobs': [1],
        'random_state': RANDOM_STATE_LIST,
    }
}

# all_params = sorted(set([k for m in methods_params_clf for k in methods_params_clf[m]]))
all_params = sorted(set([k for m in methods_params_sup for k in methods_params_sup[m]]))


def dataset_preprocessing(df, d, max_nbr_values, max_nbr_values_cat, one_hot_encode_cat, categorical_indices,
                          numerical_indices, numerical_scaler):

    nbr_classes = None
    distrib_classes = None
    avg_target = None
    std_target = None
    nbr_clusters = None
    distrib_clusters = None

    if TASK_TYPE == TASK_CLF or TASK_TYPE == TASK_CLC:
        target = dataset_target[d]
        values, counts = np.unique(df[target], return_counts=True)
        nbr_classes = len(values)
        distrib_classes = counts/len(df)
        distrib_classes = distrib_classes.tolist()
        y = df[target].values
    elif TASK_TYPE == TASK_REG or TASK_TYPE == TASK_CLR:
        target = dataset_target[d]
        avg_target = np.mean(df[target])
        std_target = np.std(df[target])
        y = df[target].values
    elif TASK_TYPE == TASK_CLU:
        target = dataset_target[d]
        values, counts = np.unique(df[target], return_counts=True)
        nbr_clusters = len(values)
        distrib_clusters = counts/len(df)
        distrib_clusters = distrib_clusters.tolist()
        y = None
    else:
        raise Exception('Unknown TASK_TYPE %s' % TASK_TYPE)

    features = [c for c in df.columns if c != target]
    res = prepare_data(df[features].values, max_nbr_values, max_nbr_values_cat,
                       df[features].columns, one_hot_encode_cat, categorical_indices,
                       numerical_indices, numerical_scaler)

    X = res[0]
    # _, is_cat_feat_r, _, is_cat_feat, _, _ = res[1]
    feature_values_r, is_cat_feat_r, feature_values, is_cat_feat, data_encoder, feature_names = res[1]

    dataset_stats = {
        'dataset': d,
        'records': df.shape[0],
        'features': df.shape[1],
        'features_onehot': len(is_cat_feat),
        'num_features': df.shape[1] - np.sum(is_cat_feat_r),
        'cat_features': np.sum(is_cat_feat_r),
        'num_features_onehot': len(is_cat_feat) - np.sum(is_cat_feat),
        'cat_features_onehot': np.sum(is_cat_feat),
        'missing_values_feat': np.sum(df.isna()).tolist(),
        'missing_values_tot': np.sum(df.isna()).sum(),
        'task_type': TASK_TYPE,
        'target': target,
        'nbr_classes': nbr_classes,
        'distrib_classes': distrib_classes,
        'avg_target': avg_target,
        'std_target': std_target,
        'nbr_clusters': nbr_clusters,
        'distrib_clusters': distrib_clusters,
    }

    return X, y, dataset_stats


def remove_missing_values(df):
    for column_name, nbr_missing in df.isna().sum().to_dict().items():
        if nbr_missing > 0:
            if column_name in df._get_numeric_data().columns:
                mean = df[column_name].mean()
                df[column_name].fillna(mean, inplace=True)
            else:
                mode = df[column_name].mode().values[0]
                df[column_name].fillna(mode, inplace=True)
    return df


def add_dataset_stats(ds):
    df_dataset_stats = pd.DataFrame(data=[ds])
    if not os.path.isfile(results_path + DATASETS_STATS_FILENAME):
        df_dataset_stats.to_csv(results_path + DATASETS_STATS_FILENAME, index=False)
    else:
        df_dataset_stats_cache = pd.read_csv(results_path + DATASETS_STATS_FILENAME)
        if df_dataset_stats['dataset'].iloc[0] not in df_dataset_stats_cache['dataset'].values:
            df_dataset_stats.to_csv(results_path + DATASETS_STATS_FILENAME, mode='a', index=False, header=False)


def evaluate_clf(y_test, y_pred, y_pred_proba):

    class_values = np.unique(y_test)
    binary = len(class_values) <= 2
    res = {
        'accuracy': skm.accuracy_score(y_test, y_pred),
        'balanced_accuracy': skm.balanced_accuracy_score(y_test, y_pred),
        'f1_score': skm.f1_score(y_test, y_pred, average='binary', pos_label=class_values[1])
        if binary else skm.f1_score(y_test, y_pred, average='micro'),
        'f1_micro': skm.f1_score(y_test, y_pred, average='micro'),
        'f1_macro': skm.f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': skm.f1_score(y_test, y_pred, average='weighted'),
        'precision_score': skm.precision_score(y_test, y_pred, average='binary', pos_label=class_values[1])
        if binary else skm.precision_score(y_test, y_pred, average='micro'),
        'precision_micro': skm.precision_score(y_test, y_pred, average='micro'),
        'precision_macro': skm.precision_score(y_test, y_pred, average='macro'),
        'precision_weighted': skm.precision_score(y_test, y_pred, average='weighted'),
        'recall_score': skm.recall_score(y_test, y_pred, average='binary', pos_label=class_values[1])
        if binary else skm.recall_score(y_test, y_pred, average='micro'),
        'recall_micro': skm.recall_score(y_test, y_pred, average='micro'),
        'recall_macro': skm.recall_score(y_test, y_pred, average='macro'),
        'recall_weighted': skm.recall_score(y_test, y_pred, average='weighted'),
        'roc_macro': skm.roc_auc_score(y_test, y_pred_proba[:, 1], average='macro') if binary else skm.roc_auc_score(
            y_test, y_pred_proba, average='macro', multi_class='ovr'),
        'roc_micro': skm.roc_auc_score(y_test, y_pred_proba[:, 1], average='micro') if binary else skm.roc_auc_score(
            y_test, y_pred_proba, average='micro', multi_class='ovr'),
        'roc_weighted': skm.roc_auc_score(y_test, y_pred_proba[:, 1], average='weighted')
        if binary else skm.roc_auc_score(y_test, y_pred_proba, average='weighted', multi_class='ovr'),
        'average_precision_macro': skm.average_precision_score(y_test, y_pred_proba[:, 1], average='macro', pos_label=class_values[1])
        if binary else skm.average_precision_score(y_test, y_pred_proba, average='macro'),
        'average_precision_micro': skm.average_precision_score(y_test, y_pred_proba[:, 1], average='micro', pos_label=class_values[1])
        if binary else skm.average_precision_score(y_test, y_pred_proba, average='micro'),
        'average_precision_weighted': skm.average_precision_score(y_test, y_pred_proba[:, 1], average='weighted', pos_label=class_values[1])
        if binary else skm.average_precision_score(y_test, y_pred_proba, average='weighted'),
    }
    return res


def evaluate_reg(y_test, y_pred):
    res = {
        'explained_variance': skm.explained_variance_score(y_test, y_pred),
        'max_error': skm.max_error(y_test, y_pred),
        'mean_absolute_error': skm.mean_absolute_error(y_test, y_pred),
        'mean_squared_error': skm.mean_squared_error(y_test, y_pred),
        'mean_squared_log_error': skm.mean_squared_log_error(y_test, y_pred),
        'median_absolute_error': skm.median_absolute_error(y_test, y_pred),
        'r2': skm.r2_score(y_test, y_pred),
        'mean_absolute_percentage_error': skm.mean_absolute_percentage_error(y_test, y_pred),
    }
    return res


def evaluate_clu(y_test, y_pred, X, dist):
    res = {
        'adjusted_mutual_info': skm.adjusted_mutual_info_score(y_test, y_pred),
        'adjusted_rand': skm.adjusted_rand_score(y_test, y_pred),
        'completeness': skm.completeness_score(y_test, y_pred),
        'fowlkes_mallows': skm.fowlkes_mallows_score(y_test, y_pred),
        'homogeneity': skm.homogeneity_score(y_test, y_pred),
        'mutual_info': skm.mutual_info_score(y_test, y_pred),
        'normalized_mutual_info': skm.normalized_mutual_info_score(y_test, y_pred),
        'rand_score': skm.rand_score(y_test, y_pred),
        'v_measure': skm.v_measure_score(y_test, y_pred),
        'silhouette_score': skm.silhouette_score(dist, y_pred),
        'calinski_harabasz': skm.calinski_harabasz_score(X, y_pred),
        'davies_bouldin': skm.davies_bouldin_score(X, y_pred)
    }
    return res


def main():
    dataset_argv = sys.argv[1]
    dataset_result_filename = RESULT_FILENAME.replace('.csv', '_%s.csv' % dataset_argv)
    print(datetime.datetime.now(), 'BENCHMARK STARTED')
    print(datetime.datetime.now(), 'Task: %s' % TASK_TYPE)
    path = dataset_path + task_folder[TASK_TYPE]
    datasets_dict = dict()
    for filename in os.listdir(path):
        if os.path.isfile(path + filename) and filename.endswith('.csv'):
            datasets_dict[filename.replace('.csv', '')] = filename

    max_nbr_values_cat = MAX_NBR_VALUES_CAT
    max_nbr_values = MAX_NBR_VALUES
    one_hot_encode_cat = ONE_HOT_ENCODE_CAT
    numerical_scaler = StandardScaler()

    df_res_cache = None
    # if os.path.isfile(results_path + RESULT_FILENAME):
    if os.path.isfile(results_path + dataset_result_filename):
        df_res_cache = pd.read_csv(results_path + dataset_result_filename, low_memory=False)
        # df_res_cache = pd.read_csv(results_path + RESULT_FILENAME)

    for dataset_name in [dataset_argv]:  #datasets_dict:
        print(datetime.datetime.now(), 'Task: %s, Dataset: %s' % (TASK_TYPE, dataset_name))
        filename = datasets_dict[dataset_name]
        df = pd.read_csv(path + filename, skipinitialspace=True)

        df.drop(dataset_feat_drop[dataset_name], axis=1, inplace=True)
        df = remove_missing_values(df)

        numerical_indices = np.where(np.in1d(df.columns.values, df._get_numeric_data().columns.values))[0]
        X, y, ds = dataset_preprocessing(df, dataset_name, max_nbr_values, max_nbr_values_cat, one_hot_encode_cat,
                                         categorical_indices=None, numerical_indices=numerical_indices,
                                         numerical_scaler=numerical_scaler)
        add_dataset_stats(ds)

        dist = None
        dist_train = None
        if TASK_TYPE == TASK_CLU:
            dist = skm.pairwise_distances(X, metric='euclidean')

        for rh in np.arange(NBR_REPEATED_HOLDOUT):
            print(datetime.datetime.now(), 'Task: %s, Dataset: %s, rep: %s' % (TASK_TYPE, dataset_name, rh))
            X_train = None
            y_train = None
            X_test = None
            y_test = None
            if TASK_TYPE == TASK_CLF or TASK_TYPE == TASK_CLC:
                X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                    stratify=y,
                                                                    test_size=0.3,
                                                                    random_state=rh)
                if TASK_TYPE == TASK_CLC:
                    dist_train = skm.pairwise_distances(X_train, metric='euclidean')

            elif TASK_TYPE == TASK_REG or TASK_TYPE == TASK_CLR:
                X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                    test_size=0.3,
                                                                    random_state=rh)
                if TASK_TYPE == TASK_CLR:
                    dist_train = skm.pairwise_distances(X_train, metric='euclidean')

            for method_name in ['RT']:  #task_method[TASK_TYPE]:
                print(datetime.datetime.now(), 'Task: %s, Dataset: %s (%s), Method: %s' % (
                    TASK_TYPE, dataset_name, rh, method_name))

                # methods_params = None
                if TASK_TYPE == TASK_CLF or TASK_TYPE == TASK_REG:
                    methods_params = methods_params_sup
                else:
                    methods_params = methods_params_unsup

                model = task_method[TASK_TYPE][method_name]
                params_prodcut = itertools.product(*methods_params[method_name].values())
                for params_vals in params_prodcut:
                    params_dict = dict()
                    for k, v in zip(methods_params[method_name].keys(), params_vals):
                        params_dict[k] = v
                    if method_name is 'RT':
                        if TASK_TYPE == TASK_CLF:
                            model_type = 'clf'
                        elif TASK_TYPE == TASK_REG:
                            model_type = 'reg'
                        elif TASK_TYPE == TASK_CLU:
                            model_type = 'clu'
                        elif TASK_TYPE == TASK_CLC:
                            model_type = 'clu'
                        elif TASK_TYPE == TASK_CLR:
                            model_type = 'clu'
                        else:
                            model_type = None
                        params_dict['model_type'] = model_type
                        params_dict['feature_names'] = None
                        params_dict['one_hot_encode_cat'] = False
                        params_dict['numerical_indices'] = numerical_indices
                        params_dict['numerical_scaler'] = None

                    res = {
                        'task': TASK_TYPE,
                        'dataset': dataset_name,
                        'method': method_name,
                        'repeated_holdout_id': rh,
                    }
                    res.update(params_dict)
                    for c in all_params:
                        if c not in res:
                            res[c] = -1
                    for c in ['feature_names', 'one_hot_encode_cat', 'numerical_indices', 'numerical_scaler', 'model_type']:
                        if c not in res:
                            res[c] = -1

                    params2hash = {k: params_dict[k] for k in params_dict if k != 'random_state'}
                    res['expid1'] = hashlib.md5(str.encode(''.join([str(s) for s in params2hash.values()]))).hexdigest()
                    params2hash = {k: params_dict[k] for k in params_dict if k not in ['random_state', 'repeated_holdout_id']}
                    res['expid2'] = hashlib.md5(str.encode(''.join([str(s) for s in params2hash.values()]))).hexdigest()
                    res['expid3'] = hashlib.md5(str.encode(''.join([str(s) for s in params_dict.values()]))).hexdigest()

                    # df_res_cache = None
                    # # if os.path.isfile(results_path + RESULT_FILENAME):
                    # if os.path.isfile(results_path + dataset_result_filename):
                    #     df_res_cache = pd.read_csv(results_path + dataset_result_filename, low_memory=False)
                    #     # df_res_cache = pd.read_csv(results_path + RESULT_FILENAME)

                    if df_res_cache is not None:
                        exp_already_run = res['expid3'] in df_res_cache['expid3'].values

                        if exp_already_run:
                            print('Already run')
                            continue

                    print(datetime.datetime.now(), 'Task: %s, Dataset: %s (%s), Method: %s, Params: %s' % (
                        TASK_TYPE, dataset_name, rh, method_name, str(params_vals)), end=' ')
                    model.set_params(**params_dict)

                    if TASK_TYPE == TASK_CLU:
                        start = time.time()
                        model.fit(X)
                        stop = time.time()
                        res_eval = {'fit_time': stop - start}
                        res_clu = evaluate_clu(y, model.labels_, dist, X)
                        res_eval.update(res_clu)

                    else:
                        start = time.time()
                        model.fit(X_train, y_train)
                        stop = time.time()
                        res_eval = {'fit_time': stop - start}

                        start = time.time()
                        y_pred = model.predict(X_test)
                        stop = time.time()
                        res_eval['predict_time'] = stop - start

                        if TASK_TYPE == TASK_CLF or TASK_TYPE == TASK_CLC:
                            y_pred_proba = model.predict_proba(X_test)
                            res_clf = evaluate_clf(y_test, y_pred, y_pred_proba)
                            res_eval.update(res_clf)
                            print('Accuracy: %.2f' % res_clf['accuracy'])
                        elif TASK_TYPE == TASK_REG or TASK_TYPE == TASK_CLR:
                            res_reg = evaluate_reg(y_test, y_pred)
                            res_eval.update(res_reg)
                            print('R2: %.2f' % res_reg['r2'])
                        if TASK_TYPE == TASK_CLC or TASK_TYPE == TASK_CLR:
                            res_clu = evaluate_clu(y_train, model.labels_, dist_train, X_train)
                            res_eval.update(res_clu)
                            print('NMI: %.2f' % res_clu['normalized_mutual_info'])

                    res.update(res_eval)

                    df_res = pd.DataFrame(data=[res])

                    # if not os.path.isfile(results_path + RESULT_FILENAME):
                    #     df_res.to_csv(results_path + RESULT_FILENAME, index=False)
                    # else:
                    #     df_res.to_csv(results_path + RESULT_FILENAME, mode='a', index=False, header=False)

                    if not os.path.isfile(results_path + dataset_result_filename):
                        df_res.to_csv(results_path + dataset_result_filename, index=False)
                    else:
                        df_res.to_csv(results_path + dataset_result_filename, mode='a', index=False, header=False)

        #             break
        #
        #         break
        #
        #     break
        #
        # break

    print(datetime.datetime.now(), 'BENCHMARK ENDED')


if __name__ == "__main__":
    main()
