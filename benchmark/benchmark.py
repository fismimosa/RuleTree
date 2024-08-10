import concurrent
import itertools
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from hashlib import md5

import numpy as np
import pandas as pd
import psutil
from progress_table import ProgressTable
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config import dataset_target_clf, task_method, TASK_CLF, DATASET_PATH, dataset_feat_drop_clf, \
    RESULTS_PATH, NBR_REPEATED_HOLDOUT, methods_params_clf, preprocessing_params, TASK_REG, dataset_target_reg, \
    dataset_feat_drop_reg, methods_params_reg, N_JOBS, TASK_CLC, TASK_CLR, methods_params_unsup, TASK_CLU, \
    dataset_target_clu, dataset_feat_drop_clu
from evaluation_utils import evaluate_clf, evaluate_expl, evaluate_reg, evaluate_clu_unsup, evaluate_clu_sup
from preprocessing_utils import remove_missing_values
from ruletree.RuleTreeBase import RuleTreeBase

import traceback

sem = multiprocessing.Semaphore(1)


class BoundedQueuePoolExecutor:
    def __init__(self, semaphore):
        self.semaphore = semaphore

    def release(self, future):
        self.semaphore.release()

    def submit(self, fn, *args, **kwargs):
        self.semaphore.acquire()
        future = super().submit(fn, *args, **kwargs)
        future.add_done_callback(self.release)
        return future


class BoundedQueueProcessPoolExecutor(BoundedQueuePoolExecutor, concurrent.futures.ProcessPoolExecutor):
    def __init__(self, *args, max_waiting_tasks=None, **kwargs):
        concurrent.futures.ProcessPoolExecutor.__init__(self, *args, **kwargs)
        if max_waiting_tasks is None:
            max_waiting_tasks = self._max_workers * 2
        elif max_waiting_tasks < 0:
            raise ValueError(f'Invalid negative max_waiting_tasks value: {max_waiting_tasks}')
        BoundedQueuePoolExecutor.__init__(self, multiprocessing.BoundedSemaphore(self._max_workers + max_waiting_tasks))


def run_clf(df_X: pd.DataFrame, y: pd.Series, model, hyper: dict,
            path, preprocessing_hyper, eval_clu):
    skf = StratifiedKFold(n_splits=NBR_REPEATED_HOLDOUT)

    X = df_X.values
    y = y.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    res = []

    try:
        for holdout_idx, (train_index, valid_index) in enumerate(skf.split(X_train, y_train)):
            X_train_holdout, X_valid = X_train[train_index], X_train[valid_index]
            y_train_holdout, y_valid = y_train[train_index], y_train[valid_index]

            model_inst = model(**hyper)
            model_inst.fit(X_train_holdout, y_train_holdout)

            y_pred = model_inst.predict(X_valid)
            y_pred_proba = model_inst.predict_proba(X_valid)

            if len(np.unique(y_valid)) != y_pred_proba.shape[1]:
                y_pred_proba = None

            res += [{f"validation_{k}": v for k, v in evaluate_clf(y_valid, y_pred, y_pred_proba).items()}]
            res[-1].update({f"validation_{k}": v for k, v in evaluate_expl(model_inst).items()})

            if eval_clu:
                res[-1].update({f"validation_{k}": v for k, v in evaluate_clu_sup(y_valid, y_pred).items()})
                res[-1].update({f"validation_{k}": v for k, v in
                                evaluate_clu_unsup(y_valid, X, pairwise_distances(X, metric="euclidean")).items()})

        model_inst = model(**hyper)
        start_time = time.time()
        model_inst.fit(X_train, y_train)
        stop_time = time.time()

        y_pred = model_inst.predict(X_test)
        y_pred_proba = model_inst.predict_proba(X_test)

        res = pd.DataFrame(res).mean().to_frame().T
        res["fit_time"] = stop_time - start_time
        for k, v in evaluate_clf(y_test, y_pred, y_pred_proba).items():
            res[f"test_{k}"] = v

        for k, v in evaluate_expl(model_inst).items():
            res[f"test_{k}"] = v

        if eval_clu:
            res[-1].update({f"test_{k}": v for k, v in evaluate_clu_sup(y_test, y_pred).items()})
            res[-1].update({f"test_{k}": v for k, v in
                            evaluate_clu_unsup(y_test, X, pairwise_distances(X, metric="euclidean")).items()})

    except Exception as e:
        traceback.print_stack()
        return "fail", e

    for k, v in preprocessing_hyper.items():
        res[k] = v

    res.to_csv(path, index=None)

    return "ok", res


def run_reg(df_X: pd.DataFrame, y: pd.Series, model, hyper: dict,
            path, preprocessing_hyper, eval_clu):
    kf = KFold(n_splits=NBR_REPEATED_HOLDOUT)

    X = df_X.values
    y = y.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    res = []

    try:
        for holdout_idx, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
            X_train_holdout, X_valid = X_train[train_index], X_train[valid_index]
            y_train_holdout, y_valid = y_train[train_index], y_train[valid_index]

            model_inst = model(**hyper)
            model_inst.fit(X_train_holdout, y_train_holdout)

            y_pred = model_inst.predict(X_valid)

            res += [{f"validation_{k}": v for k, v in evaluate_reg(y_valid, y_pred).items()}]
            res[-1].update({f"validation_{k}": v for k, v in evaluate_expl(model_inst).items()})
            if eval_clu:
                res[-1].update({f"validation_{k}": v for k, v in evaluate_clu_sup(y_test, y_pred).items()})
                res[-1].update({f"validation_{k}": v for k, v in
                                evaluate_clu_unsup(y_test, X, pairwise_distances(X, metric="euclidean")).items()})

        model_inst = model(**hyper)
        start_time = time.time()
        model_inst.fit(X_train, y_train)
        stop_time = time.time()

        y_pred = model_inst.predict(X_test)

        res = pd.DataFrame(res).mean().to_frame().T
        res["fit_time"] = stop_time - start_time
        for k, v in evaluate_reg(y_test, y_pred).items():
            res[f"test_{k}"] = v

        for k, v in evaluate_expl(model_inst).items():
            res[f"test_{k}"] = v

        if eval_clu:
            res.update({f"test_{k}": v for k, v in evaluate_clu_sup(y_test, y_pred).items()})
            res.update({f"test_{k}": v for k, v in
                        evaluate_clu_unsup(y_test, X, pairwise_distances(X, metric="euclidean")).items()})

    except Exception as e:
        traceback.print_stack()
        return "fail", e

    for k, v in preprocessing_hyper.items():
        res[k] = v

    res.to_csv(path, index=None)

    return "ok", res


def run_clu(df_X: pd.DataFrame, y: pd.Series, model, hyper: dict, path, preprocessing_hyper):
    X = df_X.values

    y_train, y_test = None, None
    if y is None:
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    else:
        y = y.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    res = dict()

    try:
        model_inst = model(**hyper)
        start_time = time.time()
        model_inst.fit(X_train, y_train)
        stop_time = time.time()

        y_pred_train = model_inst.predict(X_train)
        y_pred_test = model_inst.predict(X_test)

        res["fit_time"] = stop_time - start_time
        scores_train = evaluate_clu_unsup(y_pred_train, X_train, pairwise_distances(X_train, metric='euclidean'))
        scores_test = evaluate_clu_unsup(y_pred_test, X_test, pairwise_distances(X_test, metric='euclidean'))
        if y is not None:
            scores_train.update(evaluate_clu_sup(y_train, y_pred_train))
            scores_test.update(evaluate_clu_sup(y_test, y_pred_test))

        for k, v in scores_train.items():
            res[f"train_{k}"] = v
        for k, v in scores_test.items():
            res[f"test_{k}"] = v

        for k, v in evaluate_expl(model_inst).items():
            res[f"test_{k}"] = v

    except Exception as e:
        traceback.print_stack()
        return "fail", e

    for k, v in preprocessing_hyper.items():
        res[k] = v

    pd.DataFrame.from_dict([res]).to_csv(path, index=None)

    return "ok", res


def callback_done_clf(future,
                      table: ProgressTable,
                      dataset_name: str,
                      model_name,
                      preprocessing_hyper: dict,
                      hyper: dict):
    res_code, res = future.result()

    with sem:
        table["dataset"] = dataset_name
        table["method"] = model_name
        table.update_from_dict({k[:7]: v for k, v in preprocessing_hyper.items()})
        table.update_from_dict({k[:7]: v for k, v in hyper.items()})

        if res_code == "fail":
            table["Error"] = res
            table.next_row()
            return

        table["accuracy"] = res["test_accuracy"].iloc[0]
        table["fit_time"] = res["fit_time"].iloc[0]

        table.next_row()


def callback_done_reg(future,
                      table: ProgressTable,
                      dataset_name: str,
                      model_name,
                      preprocessing_hyper: dict,
                      hyper: dict):
    res_code, res = future.result()

    with sem:
        table["dataset"] = dataset_name
        table["method"] = model_name
        table.update_from_dict({k[:7]: v for k, v in preprocessing_hyper.items()})
        table.update_from_dict({k[:7]: v for k, v in hyper.items()})

        if res_code == "fail":
            table["Error"] = res
            table.next_row()
            return

        table["r2"] = res["test_r2"].iloc[0]
        table["fit_time"] = res["fit_time"].iloc[0]

        table.next_row()


def callback_done_clu(future,
                      table: ProgressTable,
                      dataset_name: str,
                      model_name,
                      preprocessing_hyper: dict,
                      hyper: dict):
    res_code, res = future.result()

    with sem:
        table["dataset"] = dataset_name
        table["method"] = model_name
        table.update_from_dict({k[:7]: v for k, v in preprocessing_hyper.items()})
        table.update_from_dict({k[:7]: v for k, v in hyper.items()})

        if res_code == "fail":
            table["Error"] = res
            table.next_row()
            return

        table["sil"] = res["train_silhouette_score"].iloc[0]
        table["fit_time"] = res["fit_time"].iloc[0]

        table.next_row()


def generate_filename(dataset_name, model_name, hyper, preprocessing_hyper) -> str:
    all_hyper = hyper.copy()
    all_hyper.update(preprocessing_hyper)
    all_hyper["dataset_name"] = dataset_name
    all_hyper["model_name"] = model_name

    return md5(json.dumps(all_hyper, sort_keys=True).encode()).hexdigest() + ".zip"


def benchmark(task, methods_params, dataset_target, dataset_feat_drop):
    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)
    if not os.path.exists(RESULTS_PATH + task):
        os.mkdir(RESULTS_PATH + task)

    basepath = RESULTS_PATH + task + "/"

    exp_to_skip = set([os.path.normpath(x) for x in glob(basepath+"*/*/*.zip")])

    for method_name, model in task_method[task].items():
        if not os.path.exists(basepath + method_name):
            os.mkdir(basepath + method_name)

        n_workers = psutil.cpu_count(logical=False)
        if "n_jobs" in methods_params[method_name]:
            n_workers = int(n_workers // N_JOBS)

        table = ProgressTable(pbar_embedded=False,
                              pbar_show_progress=True,
                              pbar_show_percents=True,
                              pbar_show_throughput=False,
                              num_decimal_places=3)

        print(f"===================================== n_workers = {n_workers} =====================================")

        with BoundedQueueProcessPoolExecutor(max_workers=n_workers) as exe:
            for dataset_name, target in dataset_target.items():
                print(f"===================================== dataset = {dataset_name}")
                skip_count = 0
                df = pd.read_csv(DATASET_PATH + task + "/" + dataset_name + ".csv", skipinitialspace=True)

                if not os.path.exists(basepath + method_name + "/" + dataset_name):
                    os.mkdir(basepath + method_name + "/" + dataset_name)

                if len(dataset_feat_drop[dataset_name]) > 0:
                    df.drop(dataset_feat_drop[dataset_name], axis=1, inplace=True)
                df = remove_missing_values(df)

                for one_hot_encode_cat, max_n_vals, max_n_vals_cat in itertools.product(*preprocessing_params.values()):
                    if not issubclass(model, RuleTreeBase) and not one_hot_encode_cat:
                        continue

                    df_X, ct, cat_cols_names, cont_cols_names = preprocess(df.drop(columns=[target]).copy(),
                                                                           one_hot_encode_cat,
                                                                           max_n_vals,
                                                                           max_n_vals_cat)
                    y = df[target]

                    params_prodcut = itertools.product(*methods_params[method_name].values())

                    for params_vals in list(params_prodcut):
                        params_dict = dict(zip(methods_params[method_name].keys(), params_vals))

                        filename = generate_filename(dataset_name,
                                                     method_name,
                                                     params_dict,
                                                     {"one_hot_encode_cat": one_hot_encode_cat,
                                                      "max_n_vals": max_n_vals,
                                                      "max_n_vals_cat": max_n_vals_cat})

                        path = basepath + method_name + "/" + dataset_name + "/" + filename
                        preprocessing_hyper = {"one_hot_encode_cat": one_hot_encode_cat,
                                               "max_n_vals": max_n_vals,
                                               "max_n_vals_cat": max_n_vals_cat}

                        if os.path.normpath(path) in exp_to_skip:
                            if skip_count == 10**4:
                                skip_count = 0
                                with sem:
                                    table["dataset"] = dataset_name
                                    table["method"] = method_name
                                    table["error"] = "skip10^4"
                                    table.next_row()
                            skip_count += 1
                            continue

                        if task in [TASK_CLF, TASK_CLC]:
                            process = exe.submit(run_clf,
                                                 df_X,
                                                 y,
                                                 model,
                                                 params_dict,
                                                 path,
                                                 preprocessing_hyper,
                                                 task == TASK_CLC)
                            process.add_done_callback(lambda x: callback_done_clf(
                                future=x, table=table,
                                preprocessing_hyper=preprocessing_hyper,
                                hyper=params_dict,
                                dataset_name=dataset_name,
                                model_name=method_name
                            ))
                        elif task in [TASK_REG, TASK_CLR]:
                            if TASK_CLR == task:
                                raise Exception(
                                    "Unimplemented task - Parlare con Rick di metriche in caso di regressione")

                            process = exe.submit(run_reg,
                                                 df_X,
                                                 y,
                                                 model,
                                                 params_dict,
                                                 path,
                                                 preprocessing_hyper,
                                                 task == TASK_CLR)
                            process.add_done_callback(lambda x: callback_done_reg(
                                future=x, table=table,
                                preprocessing_hyper=preprocessing_hyper,
                                hyper=params_dict,
                                dataset_name=dataset_name,
                                model_name=method_name
                            ))
                        elif task == TASK_CLU:
                            process = exe.submit(run_clu,
                                                 df_X,
                                                 y,
                                                 model,
                                                 params_dict,
                                                 path,
                                                 preprocessing_hyper)
                            process.add_done_callback(lambda x: callback_done_clu(
                                future=x, table=table,
                                preprocessing_hyper=preprocessing_hyper,
                                hyper=params_dict,
                                dataset_name=dataset_name,
                                model_name=method_name
                            ))


def preprocess(df: pd.DataFrame, one_hot_encode_cat: bool, max_n_vals, max_n_vals_cat):
    cat_cols_names = []
    cont_cols_names = []

    for col_idx, (col_name, col_dtype) in enumerate(zip(df.columns, df.dtypes)):
        if col_dtype == np.dtype('O'):
            cat_cols_names.append(col_name)
            continue

        if len(np.unique(df[col_name])) > max_n_vals:
            _, vals = np.histogram(values, bins=max_n_vals)
            values = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
            df[col_name] = values

        if len(np.unique(df[col_name])) <= max_n_vals_cat:
            cat_cols_names.append(col_name)
            df[col_name] = df[col_name].apply(lambda x: f"{col_name}_is_{x}")
        else:
            cont_cols_names.append(col_name)

    if one_hot_encode_cat:
        ct = ColumnTransformer(
            [("cat", OneHotEncoder(), make_column_selector(dtype_include="object")),
             ('std_scaler', StandardScaler(), make_column_selector(dtype_include=['int', 'float']))],
            remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=1)
    else:
        ct = ColumnTransformer(
            [('std_scaler', StandardScaler(), make_column_selector(dtype_include=['int', 'float']))],
            remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=1)

    return pd.DataFrame(ct.fit_transform(df), columns=ct.get_feature_names_out()), ct, cat_cols_names, cont_cols_names


def main():
    if len(sys.argv) == 1:
        print("Please provide at least one argument in ['CLF', 'REG', 'CLU', 'CLC', 'CLR']")

    for arg in sys.argv[1:]:
        if arg == TASK_CLF:
            benchmark(task=TASK_CLF,
                      methods_params=methods_params_clf,
                      dataset_target=dataset_target_clf,
                      dataset_feat_drop=dataset_feat_drop_clf)

        elif arg == TASK_REG:
            benchmark(task=TASK_REG,
                      methods_params=methods_params_reg,
                      dataset_target=dataset_target_reg,
                      dataset_feat_drop=dataset_feat_drop_reg)

        elif arg == "CLU":
            benchmark(task=TASK_CLU,
                      methods_params=methods_params_unsup,
                      dataset_target=dataset_target_clu,
                      dataset_feat_drop=dataset_feat_drop_clu)

        elif arg == TASK_CLC:
            benchmark(task=TASK_CLC,
                      methods_params=methods_params_unsup,
                      dataset_target=dataset_target_clf,
                      dataset_feat_drop=dataset_feat_drop_clf)

        elif arg == TASK_CLR:
            benchmark(task=TASK_CLR,
                      methods_params=methods_params_unsup,
                      dataset_target=dataset_target_reg,
                      dataset_feat_drop=dataset_feat_drop_reg)

        else:
            print(f"Unknown argument: {arg}")


if __name__ == '__main__':
    main()
