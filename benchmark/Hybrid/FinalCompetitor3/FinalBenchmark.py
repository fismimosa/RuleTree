import math
import os
import pickle

os.environ['OPENBLAS_NUM_THREADS'] = '32'
os.environ["OMP_NUM_THREADS"] = '32'
os.environ["OPENBLAS_NUM_THREADS"] = '32'
os.environ["MKL_NUM_THREADS"] = '32'
os.environ["NUMEXPR_NUM_THREADS"] = '32'
os.environ["VECLIB_MAXIMUM_THREADS"] = '32'

from interpret.glassbox import ExplainableBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import ExtraTreeClassifier

from benchmark.Hybrid2.FinalCompetitor3.HyperFinal import get_hyperparameters, n_jobs

from numba import UnsupportedError
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, MaxAbsScaler

import traceback
import warnings

from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier, HistGradientBoostingClassifier

from RuleTree.stumps.classification import DecisionTreeStumpClassifier, ObliqueDecisionTreeStumpClassifier, \
    PivotTreeStumpClassifier, MultiplePivotTreeStumpClassifier, ObliquePivotTreeStumpClassifier
from RuleTree.stumps.classification.MultipleObliquePivotTreeStumpClassifier import \
    MultipleObliquePivotTreeStumpClassifier
from RuleTree.stumps.classification.PartialPivotTreeStumpClassifier import PartialPivotTreeStumpClassifier
from RuleTree.stumps.classification.PartialProximityTreeStumpClassifier import PartialProximityTreeStumpClassifier
from benchmark.Hybrid2.HybridReaders import *
from xgboost import XGBClassifier

import hashlib
import os.path
import time
from concurrent.futures import ProcessPoolExecutor
from copy import copy

import pandas as pd
import psutil
from tqdm.auto import tqdm

from RuleTree import RuleTreeClassifier, RuleForestClassifier
from benchmark.evaluation_utils import evaluate_expl, evaluate_clf

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=Warning)

scalers = {
    'None': None,
    #'StandardScaler': StandardScaler,
    #'MinMaxScaler': MinMaxScaler,
    #'MaxAbsScaler': MaxAbsScaler,
    #'RobustScaler': RobustScaler,
}


def compute_measures(y_test, y_pred, model):
    measures = evaluate_clf(y_test, y_pred, y_pred_proba=None)
    measures |= evaluate_expl(model)

    return measures



def run(method, hyper_dict, hyper_stump_dict, dataset_name, df: pd.DataFrame):
    use_combination = False
    if 'ONLY_PartialProximity' != method:
        return pd.DataFrame()
    
    hyper_dict = copy(hyper_dict)

    filename_hash = '|'.join([
        dataset_name,
        method,
        hashlib.md5('|'.join([f'{x}' for x in hyper_dict.values()]).encode()).hexdigest(),
        hashlib.md5('|'.join([f'{x}' for x in hyper_stump_dict.values()]).encode()).hexdigest(),
    ]) + '.csv'
    if os.path.exists(f"res_tmp/{dataset_name}/{filename_hash}"):
        return pd.read_csv(f"res_tmp/{dataset_name}/{filename_hash}")

    if not use_combination:
        filename_hash = '|'.join([
            dataset_name,
            method,
            hashlib.md5('|'.join([f'{x}' for x in hyper_dict.values()]).encode()).hexdigest(),
            hashlib.md5('|'.join([f'{x}' for x in hyper_stump_dict.values()]).encode()).hexdigest(),
        ]) + '_noCOMB.csv'
        if os.path.exists(f"res_tmp/{dataset_name}/{filename_hash}"):
            return pd.read_csv(f"res_tmp/{dataset_name}/{filename_hash}")

    scores = copy(hyper_dict) | {'method': method}
    scores |= copy(hyper_stump_dict)

    y = LabelEncoder().fit_transform(df[df.columns[-1]].values)
    X = df[df.columns[:-1]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scores_kv = []
    skf = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    hyper_dict_original = copy(hyper_dict)
    for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        hyper_dict = copy(hyper_dict_original)
        X_train_kf, X_val, y_train_kf, y_val = (X_train[train_index], X_train[test_index],
                                                y_train[train_index], y_train[test_index])
        if method == "DT":
            model = RuleTreeClassifier(**hyper_dict)
        elif method == "RF":
            model = RandomForestClassifier(**hyper_dict)
        elif method == "AB":
            model = AdaBoostClassifier(**hyper_dict)
        elif method == 'ETC':
            model = ExtraTreeClassifier(**hyper_dict)
        elif method == "ETsC":
            model = ExtraTreesClassifier(**hyper_dict)
        elif method == "GBC":
            model = GradientBoostingClassifier(**hyper_dict)
        elif method == "HGBC":
            model = HistGradientBoostingClassifier(**hyper_dict)
        elif method == "XGB":
            model = XGBClassifier(**hyper_dict)
        elif method == 'Dart':
            model = XGBClassifier(**hyper_dict)
        elif method == 'LGBM':
            model = LGBMClassifier(**hyper_dict, verbosity=-1)
        elif method == 'EBM':
            model = ExplainableBoostingClassifier(**hyper_dict)
        else:
            base_stumps = []
            if 'base_stumps' not in hyper_dict:
                raise ValueError(f"Base stumps not found in hyper_dict for {method}")
            for stump in hyper_dict["base_stumps"]:
                if stump == "PartialPivotTreeStumpClassifier":
                    base_stumps.append(PartialPivotTreeStumpClassifier(**hyper_stump_dict[stump], use_combination=use_combination))
                    base_stumps.append(PartialPivotTreeStumpClassifier(**hyper_stump_dict[stump], scaler=MinMaxScaler(), use_combination=use_combination))
                    base_stumps.append(PartialPivotTreeStumpClassifier(**hyper_stump_dict[stump], scaler=StandardScaler(), use_combination=use_combination))
                elif stump == "PartialProximityTreeStumpClassifier":
                     base_stumps.append(PartialProximityTreeStumpClassifier(**hyper_stump_dict[stump], use_combination=use_combination))
                     base_stumps.append(PartialProximityTreeStumpClassifier(**hyper_stump_dict[stump], scaler=MinMaxScaler(), use_combination=use_combination))
                     base_stumps.append(PartialProximityTreeStumpClassifier(**hyper_stump_dict[stump], scaler=StandardScaler(), use_combination=use_combination))
                elif stump == "ObliqueDecisionTreeStumpClassifier":
                    base_stumps.append(ObliqueDecisionTreeStumpClassifier(**hyper_stump_dict[stump]))
                elif stump == "ObliquePivotTreeStumpClassifier":
                    base_stumps.append(ObliquePivotTreeStumpClassifier(**hyper_stump_dict[stump]))
                elif stump == "PivotTreeStumpClassifier":
                    base_stumps.append(PivotTreeStumpClassifier(**hyper_stump_dict[stump]))
                elif stump == "MultiplePivotTreeStumpClassifier":
                    base_stumps.append(MultiplePivotTreeStumpClassifier(**hyper_stump_dict[stump]))
                elif stump == "MultipleObliquePivotTreeStumpClassifier":
                    base_stumps.append(MultipleObliquePivotTreeStumpClassifier(**hyper_stump_dict[stump]))
                elif stump == "DecisionTreeStumpClassifier":
                    try:
                        base_stumps.append(DecisionTreeStumpClassifier(**hyper_stump_dict[stump]))
                    except KeyError as e:
                        raise e
                else:
                    raise ValueError(f"Unknown stump type: {stump}")

            del hyper_dict["base_stumps"]

            if method == 'RFHDT':
                _n_estimators = hyper_dict['n_estimators']
                _criterion = hyper_dict['criterion']
                _n_jobs = hyper_dict['n_jobs']

                del hyper_dict['n_estimators']
                del hyper_dict['criterion']

                model = RuleForestClassifier(
                    n_estimators=_n_estimators, criterion=_criterion, n_jobs=1, distance_measure='euclidean',
                    custom_estimator=RuleTreeClassifier(**hyper_dict),  # foresta di alberi ibridi
                )
            elif method == 'HRFDT':
                _n_estimators = hyper_dict['n_estimators']
                _criterion = hyper_dict['criterion']

                del hyper_dict['n_estimators']
                del hyper_dict['criterion']
                del hyper_dict['n_jobs']

                model = RuleForestClassifier(
                    n_estimators=_n_estimators, criterion=_criterion, n_jobs=1, distance_measure='euclidean',
                    base_stumps=base_stumps,  # base_stumps #foresta ibrida di alberi omogenei
                )
            else:
                model = RuleTreeClassifier(**hyper_dict, base_stumps=base_stumps)

        print(f'[START]{dataset_name}-{method} CV {i} \t{filename_hash}')
        start = time.time()
        try:
            model.fit(X_train_kf, y_train_kf)
        except UnsupportedError:
            traceback.print_exc()
            return pd.DataFrame()
        except Exception as e:
            traceback.print_exc()
            return pd.DataFrame()
        end = time.time()

        y_pred_val = model.predict(X_val)

        dict_res = {f'{k}_val': v for k, v in compute_measures(y_val, y_pred_val, model).items()}
        scores_kv.append(dict_res)

        print(f'[FINE]{dataset_name}-{method} CV {i} \t{filename_hash} {int((end-start)/60)}min')

    hyper_dict = hyper_dict_original.copy()

    if method == "DT":
        model = RuleTreeClassifier(**hyper_dict)
    elif method == "RF":
        model = RandomForestClassifier(**hyper_dict)
    elif method == "AB":
        model = AdaBoostClassifier(**hyper_dict)
    elif method == 'ETC':
        model = ExtraTreeClassifier(**hyper_dict)
    elif method == "ETsC":
        model = ExtraTreesClassifier(**hyper_dict)
    elif method == "GBC":
        model = GradientBoostingClassifier(**hyper_dict)
    elif method == "HGBC":
        model = HistGradientBoostingClassifier(**hyper_dict)
    elif method == "XGB":
        model = XGBClassifier(**hyper_dict)
    elif method == 'Dart':
        model = XGBClassifier(**hyper_dict)
    elif method == 'LGBM':
        model = LGBMClassifier(**hyper_dict)
    elif method == 'EBM':
        model = ExplainableBoostingClassifier(**hyper_dict)
    else:
        base_stumps = []
        if 'base_stumps' not in hyper_dict:
            raise ValueError(f"Base stumps not found in hyper_dict for {method}")
        for stump in hyper_dict["base_stumps"]:
            if stump == "PartialPivotTreeStumpClassifier":
                base_stumps.append(PartialPivotTreeStumpClassifier(**hyper_stump_dict[stump], use_combination=use_combination))
                base_stumps.append(PartialPivotTreeStumpClassifier(**hyper_stump_dict[stump], scaler=MinMaxScaler(), use_combination=use_combination))
                base_stumps.append(PartialPivotTreeStumpClassifier(**hyper_stump_dict[stump], scaler=StandardScaler(), use_combination=use_combination))
            elif stump == "PartialProximityTreeStumpClassifier":
                base_stumps.append(PartialProximityTreeStumpClassifier(**hyper_stump_dict[stump], use_combination=use_combination))
                base_stumps.append(
                    PartialProximityTreeStumpClassifier(**hyper_stump_dict[stump], scaler=MinMaxScaler(), use_combination=use_combination))
                base_stumps.append(
                    PartialProximityTreeStumpClassifier(**hyper_stump_dict[stump], scaler=StandardScaler(), use_combination=use_combination))
            elif stump == "ObliqueDecisionTreeStumpClassifier":
                base_stumps.append(ObliqueDecisionTreeStumpClassifier(**hyper_stump_dict[stump]))
            elif stump == "ObliquePivotTreeStumpClassifier":
                base_stumps.append(ObliquePivotTreeStumpClassifier(**hyper_stump_dict[stump]))
            elif stump == "PivotTreeStumpClassifier":
                base_stumps.append(PivotTreeStumpClassifier(**hyper_stump_dict[stump]))
            elif stump == "MultiplePivotTreeStumpClassifier":
                base_stumps.append(MultiplePivotTreeStumpClassifier(**hyper_stump_dict[stump]))
            elif stump == "MultipleObliquePivotTreeStumpClassifier":
                base_stumps.append(MultipleObliquePivotTreeStumpClassifier(**hyper_stump_dict[stump]))
            elif stump == "DecisionTreeStumpClassifier":
                base_stumps.append(DecisionTreeStumpClassifier(**hyper_stump_dict[stump]))
            else:
                raise ValueError(f"Unknown stump type: {stump}")

        del hyper_dict["base_stumps"]
        if method == 'RFHDT':
            _n_estimators = hyper_dict['n_estimators']
            _criterion = hyper_dict['criterion']
            _n_jobs = hyper_dict['n_jobs']

            del hyper_dict['n_estimators']
            del hyper_dict['criterion']

            model = RuleForestClassifier(
                n_estimators=_n_estimators, criterion=_criterion, n_jobs=1, distance_measure='euclidean',
                custom_estimator=RuleTreeClassifier(**hyper_dict),  # foresta di alberi ibridi
            )
        elif method == 'HRFDT':
            _n_estimators = hyper_dict['n_estimators']
            _criterion = hyper_dict['criterion']

            del hyper_dict['n_estimators']
            del hyper_dict['criterion']
            del hyper_dict['n_jobs']

            model = RuleForestClassifier(
                n_estimators=_n_estimators, criterion=_criterion, n_jobs=1, distance_measure='euclidean',
                base_stumps=base_stumps,  # base_stumps #foresta ibrida di alberi omogenei
            )
        else:
            model = RuleTreeClassifier(**hyper_dict, base_stumps=base_stumps)

    try:
        process = psutil.Process(os.getpid())
        before = process.memory_info().rss
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        after = process.memory_info().rss

        before_pred = process.memory_info().rss
        start_pred = time.time()
        y_pred = model.predict(X_test)
        end_pred = time.time()
        after_pred = process.memory_info().rss
    except Exception as e:
        traceback.print_exc()
        return pd.DataFrame()

    scores |= (
            {'train_time': end - start, 'pred_time': end_pred - start_pred}
            | {'train_mem_MB': (after - before) / 1024 ** 2, 'pred_mem_MB': (after_pred - before_pred) / 1024 ** 2}
            | {f'{k}_test': v for k, v in compute_measures(y_test, y_pred, model).items()}
    )

    scores['USE_COMB'] = use_combination

    df_val = pd.DataFrame(scores_kv).mean().to_frame().T
    df_val_std = pd.DataFrame(scores_kv).std().to_frame().T

    df = pd.DataFrame.from_dict([scores])
    df[df_val.columns] = df_val
    df[[f'{x}_std' for x in df_val_std.columns]] = df_val
    try:
        df.to_csv(f"res_tmp/{dataset_name}/{filename_hash}", index=False)
        pickle.dump(model, open(f"res_tmp/{dataset_name}/{filename_hash.replace('.csv', '.pickle')}", "wb"))
    except Exception as e:
        raise e

    return df


def main():
    datasets = dict() #small_datasets
    for dataset_fun in [read_ionosphere, read_wdbc]:#(small_datasets+medium_datasets+big_datasets)[:-2]:
        dataset_name, df = dataset_fun()

        if dataset_name in ['sonar']:
            continue
        print(dataset_name, df.shape)
        for scaler_name, scaler_fun in scalers.items():
            if scaler_fun is None:
                datasets[dataset_name] = df
            else:
                df_scaled = pd.DataFrame(scaler_fun().fit_transform(df.iloc[:, :-1]).reshape(df.shape[0], -1),
                                         columns=df.columns[:-1])
                df_scaled[df.columns[-1]] = df.iloc[:, -1]
                datasets[f'{dataset_name}_{scaler_name}'] = df_scaled


    n_process = int(psutil.cpu_count(logical=True) / n_jobs)
    print(n_process)
    n_process = 8

    processes = []
    dataframes = []
    single_t = False

    with ProcessPoolExecutor(max_workers=n_process) as executor:
        for dataset_name, df in datasets.items():
            if not os.path.exists("res_tmp/"):
                os.mkdir("res_tmp/")
            if not os.path.exists("res_tmp/" + dataset_name):
                os.mkdir("res_tmp/" + dataset_name)

            #processes = []
            #dataframes = []
            #single_t = False

            if single_t:
                for i, (model, hyper_dict, hyper_stump_dict) in enumerate(tqdm(list(get_hyperparameters(df)),
                                                                desc=f"{dataset_name}: submitting experiments")):

                    print(model)
                    #tqdm.write(f'model: {model} {hyper_dict}\r\n\t{hyper_stump_dict}')
                    processes.append(run(model, hyper_dict, hyper_stump_dict, dataset_name, df))
                    #tqdm.write('ENDED\r\n\r\n')

                for process in tqdm(processes, desc=f"{dataset_name}: collecting results"):
                    dataframes.append(process)

            else:
                for model, hyper_dict, hyper_stump_dict in tqdm(list(get_hyperparameters(df)),
                                                                desc=f"{dataset_name}: submitting experiments"):
                    processes.append((dataset_name,
                                      executor.submit(run, model, hyper_dict, hyper_stump_dict, dataset_name, df)))

                """for _, process in tqdm(processes, desc=f"{dataset_name}: collecting results"):
                    try:
                        res = process.result()
                        dataframes.append(res)
                    except Exception as e:
                        traceback.print_exc()"""


        it = tqdm(processes)
        for dataset_name, process in it:
            try:
                it.set_description(f"{dataset_name}: collecting results")
                res = process.result()
                dataframes.append(res)
            except Exception as e:
                traceback.print_exc()

        #pd.concat(dataframes, ignore_index=True).to_csv(f"res_tmp/hybrid_benchmark_{dataset_name}.csv", index=False)


if __name__ == '__main__':
    main()
