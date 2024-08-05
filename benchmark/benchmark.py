# Codice per il benchmark salvando anche il modello e l'output dei vari metodi
import getopt
import hashlib
import itertools
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm
import sklearn.metrics as skm

from ruletree.utils.data_utils import prepare_data
from evaluation_utils import evaluate_clu_sup, evaluate_clu_unsup, evaluate_reg, evaluate_clf
from preprocessing_utils import remove_missing_values
from config import TASK_CLF, TASK_CLC, TASK_REG, TASK_CLR, TASK_CLU, dataset_target_clf, dataset_target_reg, \
    dataset_target_clu, RESULTS_PATH, DATASET_PATH, dataset_feat_drop_clf, dataset_feat_drop_reg, dataset_feat_drop_clu, \
    methods_params_sup, methods_params_unsup, MAX_NBR_VALUES, MAX_NBR_VALUES_CAT, ONE_HOT_ENCODE_CAT, \
    NBR_REPEATED_HOLDOUT, task_method, datasets_target_clu_sup


class Benchmark:
    def __init__(self, task, dataset, numerical_scaler=StandardScaler(), verbose=2):
        if task not in [TASK_CLF, TASK_CLC, TASK_REG, TASK_CLR, TASK_CLU]:
            raise ValueError(f'Unknown task {task}')

        self.task = task
        self.dataset = dataset
        self.numerical_scaler = numerical_scaler
        self.verbose = verbose  # 0 = None, 1: only tqdm, 2: tqdm+print
        self.print = lambda *arg, **kwargs: tqdm.write(" ".join([str(x) for x in arg]), **kwargs) if self.verbose > 1 \
            else None

        self.dataset_stats_path = '%s_datasets_stats.csv' % task
        self.result_filename = '%s_results_%s.csv' % (task, dataset)
        self.pickle_path = RESULTS_PATH + self.result_filename[:-4] + "/"
        os.makedirs(self.pickle_path, exist_ok=True)

        if task == TASK_CLF:
            self.dataset_feat_drop = dataset_feat_drop_clf
            self.all_params = sorted(set([k for m in methods_params_sup for k in methods_params_sup[m]]))
            self.methods_params = methods_params_sup
        elif task == TASK_REG:
            self.dataset_feat_drop = dataset_feat_drop_reg
            self.all_params = sorted(set([k for m in methods_params_sup for k in methods_params_sup[m]]))
            self.methods_params = methods_params_sup
        else:
            self.dataset_feat_drop = dataset_feat_drop_clu
            self.all_params = sorted(set([k for m in methods_params_unsup for k in methods_params_unsup[m]]))
            self.methods_params = methods_params_unsup

        self.df_res_cache = None
        if os.path.isfile(RESULTS_PATH + self.result_filename):
            self.df_res_cache = pd.read_csv(RESULTS_PATH + self.result_filename, low_memory=False)

    def dataset_preprocessing(self, df, d, max_nbr_values, max_nbr_values_cat, one_hot_encode_cat, categorical_indices,
                              numerical_indices, numerical_scaler):
        nbr_classes = None
        distrib_classes = None
        avg_target = None
        std_target = None
        nbr_clusters = None
        distrib_clusters = None

        if self.task in [TASK_CLF, TASK_CLC]:
            target = dataset_target_clf[d]
            values, counts = np.unique(df[target], return_counts=True)
            nbr_classes = len(values)
            distrib_classes = counts / len(df)
            distrib_classes = distrib_classes.tolist()
            y = df[target].values
        elif self.task in [TASK_REG, TASK_CLR]:
            target = dataset_target_reg[d]
            avg_target = np.mean(df[target])
            std_target = np.std(df[target])
            y = df[target].values
        elif self.task == TASK_CLU:
            target = dataset_target_clu[d]
            values, counts = np.unique(df[target], return_counts=True)
            nbr_clusters = len(values)
            distrib_clusters = counts / len(df)
            distrib_clusters = distrib_clusters.tolist()
            y = df[target].values
        else:
            raise Exception('Unknown TASK_TYPE %s' % self.task)

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
            'task_type': self.task,
            'target': target,
            'nbr_classes': nbr_classes,
            'distrib_classes': distrib_classes,
            'avg_target': avg_target,
            'std_target': std_target,
            'nbr_clusters': nbr_clusters,
            'distrib_clusters': distrib_clusters,
        }

        return X, y, dataset_stats

    def add_dataset_stats(self, ds):
        df_dataset_stats = pd.DataFrame(data=[ds])

        if not os.path.isfile(RESULTS_PATH + self.dataset_stats_path):
            df_dataset_stats.to_csv(RESULTS_PATH + self.dataset_stats_path, index=False)
        else:
            df_dataset_stats_cache = pd.read_csv(RESULTS_PATH + self.dataset_stats_path)
            if df_dataset_stats['dataset'].iloc[0] not in df_dataset_stats_cache['dataset'].values:
                df_dataset_stats.to_csv(RESULTS_PATH + self.dataset_stats_path, mode='a', index=False, header=False)

    def run(self):
        with threadpool_limits(limits=1):
            self.print(datetime.now(), 'BENCHMARK STARTED')
            self.print(datetime.now(), 'Task: %s' % self.task)
            df = pd.read_csv(DATASET_PATH + self.task + "/" + self.dataset + ".csv", skipinitialspace=True)

            if len(self.dataset_feat_drop[self.dataset]) > 0:
                df.drop(self.dataset_feat_drop[self.dataset], axis=1, inplace=True)
            df = remove_missing_values(df)

            numerical_indices = np.where(np.in1d(df.columns.values, df._get_numeric_data().columns.values))[0]
            X, y, ds = self.dataset_preprocessing(df, self.dataset, MAX_NBR_VALUES, MAX_NBR_VALUES_CAT,
                                                  ONE_HOT_ENCODE_CAT, categorical_indices=None,
                                                  numerical_indices=numerical_indices,
                                                  numerical_scaler=self.numerical_scaler)
            self.add_dataset_stats(ds)

            for holdout_idx in range(NBR_REPEATED_HOLDOUT):
                self.print(datetime.now(),
                           f'Task: {self.task}, Dataset: {self.dataset}, rep: {holdout_idx}/{NBR_REPEATED_HOLDOUT}')
                X_train, y_train, X_test, y_test = None, None, None, None

                dist = None
                dist_train = None
                if self.task in [TASK_CLF, TASK_CLC]:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3,
                                                                        random_state=holdout_idx)
                elif self.task in [TASK_REG, TASK_CLR]:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                                        random_state=holdout_idx)
                else:
                    dist = skm.pairwise_distances(X, metric='euclidean')

                if self.task in [TASK_CLC, TASK_CLR]:
                    dist_train = skm.pairwise_distances(X_train, metric='euclidean')

                for method_name in task_method[self.task]:
                    self.print(datetime.now(), f'Task: {self.task}, Dataset: {self.dataset}, '
                                               f'rep: {holdout_idx}/{NBR_REPEATED_HOLDOUT}, Method: {method_name}')

                    params_prodcut = itertools.product(*self.methods_params[method_name].values())

                    for params_vals in tqdm(list(params_prodcut), leave=False, position=0, disable=self.verbose == 0):
                        params_dict = dict()
                        for k, v in zip(self.methods_params[method_name].keys(), params_vals):
                            params_dict[k] = v

                        res = {  # nel codice di rick è res
                            'task': self.task,
                            'dataset': self.dataset,
                            'method': method_name,
                            'repeated_holdout_id': holdout_idx,
                        }
                        res.update(params_dict)

                        if method_name == 'RT':
                            res['feature_names'] = None
                            res['one_hot_encode_cat'] = False
                            res['numerical_indices'] = numerical_indices
                            res['numerical_scaler'] = None

                        #if some params are not used by this method, we add -1
                        for c in self.all_params:
                            if c not in params_dict:
                                res[c] = -1
                        for c in ['feature_names', 'one_hot_encode_cat', 'numerical_indices', 'numerical_scaler']:
                            if c not in params_dict:
                                res[c] = -1

                        res['exp_id'] = hashlib.md5(
                            str.encode(''.join([str(s) for s in params_dict.values()]))).hexdigest()

                        params2hash = {k: params_dict[k] for k in params_dict if k != 'random_state'}
                        res['exp_id1'] = hashlib.md5(
                            str.encode(''.join([str(s) for s in params2hash.values()]))).hexdigest()

                        params2hash = {k: params_dict[k] for k in params_dict if k not in ['random_state',
                                                                                           'repeated_holdout_id']}
                        res['exp_id2'] = hashlib.md5(
                            str.encode(''.join([str(s) for s in params2hash.values()]))).hexdigest()

                        if self.df_res_cache is not None:
                            exp_already_run = res['exp_id'] in self.df_res_cache['exp_id'].values

                            if exp_already_run:
                                self.print('Already run')
                                continue

                        self.print(datetime.now(), f'Task: {self.task}, Dataset: {self.dataset}, '
                                                   f'rep: {holdout_idx}, Method: {method_name}, '
                                                   f'Params: {str(params_vals)}', end=' ')

                        model = task_method[self.task][method_name](**params_dict)

                        run_result = None
                        y = None
                        if self.task == TASK_CLU:
                            run_result, y = self.__run_clu(X, y, model, dist)
                            if run_result is None:
                                continue
                        elif self.task == TASK_CLF:
                            run_result, y = self.__run_clf(X_train, y_train, X_test, y_test, model, dist_train)
                            if run_result is None:
                                continue
                        if self.task == TASK_REG:
                            run_result, y = self.__run_reg(X_train, y_train, X_test, y_test, model, dist_train)
                            if run_result is None:
                                continue

                        res.update(run_result)


                        # save the model
                        with open(self.pickle_path + f"{res['exp_id']}_{method_name}_model.pkl", 'wb') as f:
                            pickle.dump(model, f)
                        # save the output
                        with open(self.pickle_path + f"{res['exp_id']}_{method_name}_results.pkl", 'wb') as f:
                            pickle.dump(y, f)

                        res = dict(sorted(res.items(), key=lambda x: x[0]))
                        df_res = pd.DataFrame(data=[res])

                        if not os.path.isfile(RESULTS_PATH + self.result_filename):
                            df_res.to_csv(RESULTS_PATH + self.result_filename, index=False)
                        else:
                            df_res.to_csv(RESULTS_PATH + self.result_filename, mode='a', index=False, header=False)

        self.print(datetime.now(), 'BENCHMARK ENDED')

    def __run_clu(self, X, y, model, dist):
        try:
            start = time.time()
            model.fit(X)
            stop = time.time()
        except ValueError as e:
            print("\n", e)
            return None, None
        res_eval = {'fit_time': stop - start}
        if self.dataset in datasets_target_clu_sup:
            res_clu = evaluate_clu_sup(y, model.labels_, dist, X)
        else:
            res_clu = evaluate_clu_unsup(model.labels_, dist, X)

        self.print('Silhouette: %.2f' % res_clu['silhouette_score'])

        res_eval.update(res_clu)

        return res_eval, model.labels_

    def __run_clf(self, X_train, y_train, X_test, y_test, model, dist_train):
        try:
            start = time.time()
            model.fit(X_train, y_train)
            stop = time.time()
        except ValueError as e:
            print("\n", e)
            return None, None
        res_eval = {'fit_time': stop - start}

        start = time.time()
        y_pred = model.predict(X_test)
        stop = time.time()
        res_eval['predict_time'] = stop - start

        y_pred_proba = model.predict_proba(X_test)
        res_clf = evaluate_clf(y_test, y_pred, y_pred_proba)
        res_eval.update(res_clf)
        self.print('Accuracy: %.2f' % res_clf['accuracy'])

        if self.task == TASK_CLC:
            res_clu = evaluate_clu_sup(y_train, model.labels_, dist_train, X_train)
            res_eval.update(res_clu)

        return res_eval, y_pred

    def __run_reg(self, X_train, y_train, X_test, y_test, model, dist_train):
        try:
            start = time.time()
            model.fit(X_train, y_train)
            stop = time.time()
        except ValueError as e:
            print("\n", e)
            return None, None
        res_eval = {'fit_time': stop - start}

        start = time.time()
        y_pred = model.predict(X_test)
        stop = time.time()
        res_eval['predict_time'] = stop - start

        res_reg = evaluate_reg(y_test, y_pred)
        res_eval.update(res_reg)
        self.print('R2: %.2f' % res_reg['r2'])

        if self.task == TASK_CLR:
            res_clu = evaluate_clu_sup(y_train, model.labels_, dist_train, X_train)
            res_eval.update(res_clu)

        return res_eval, y_pred


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "t:d:", ["task=", "dataset="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)

    TASK_TYPE = ""
    DATASET_NAME = ""
    for opt, arg in opts:
        if opt in ["-t", "--task"]:
            TASK_TYPE = arg
        elif opt in ["-d", "--dataset"]:
            DATASET_NAME = arg

    Benchmark(TASK_TYPE, DATASET_NAME).run()


if __name__ == '__main__':
    main(sys.argv[1:])
