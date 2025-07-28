import hashlib
import os
import pickle
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from copy import copy

import numpy as np
import pandas as pd
import psutil
from aeon.classification.convolution_based import Arsenal, HydraClassifier, MultiRocketHydraClassifier, \
    RocketClassifier, MiniRocketClassifier, MultiRocketClassifier
from aeon.classification.deep_learning import MLPClassifier, InceptionTimeClassifier
from aeon.classification.dictionary_based import BOSSEnsemble, MrSEQLClassifier, MrSQMClassifier, MUSE, WEASEL, WEASEL_V2
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier, ProximityForest, ProximityTree
from aeon.classification.feature_based import Catch22Classifier
from aeon.classification.hybrid import HIVECOTEV1, HIVECOTEV2
from aeon.classification.interval_based import CanonicalIntervalForestClassifier, DrCIFClassifier, \
    IntervalForestClassifier
from aeon.transformations.collection.dictionary_based import BORF
from keras.src.applications.resnet import ResNet
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from tqdm.auto import tqdm

from RuleTree import RuleTreeClassifier
from RuleTree.stumps.classification import DecisionTreeStumpClassifier, ShapeletTreeStumpClassifier
from RuleTree.utils.data_utils import preprocessing
from RuleTree.utils.shapelet_transform.Shapelets import Shapelets
from benchmark.GenTSTree.GenTSBenchmark.GenTSBenchmark_Hyper import get_hyperparameters, n_jobs
from benchmark.evaluation_utils import evaluate_clf

os.environ['OPENBLAS_NUM_THREADS'] = '32'
os.environ["OMP_NUM_THREADS"] = '32'
os.environ["OPENBLAS_NUM_THREADS"] = '32'
os.environ["MKL_NUM_THREADS"] = '32'
os.environ["NUMEXPR_NUM_THREADS"] = '32'
os.environ["VECLIB_MAXIMUM_THREADS"] = '32'

from benchmark.GenTSTree.GenTSReaders import small_datasets, medium_datasets, big_datasets

scalers = {
    "none": None,
}


def _get_model(method, hyper_dict):
    hyper_dict = copy(hyper_dict)

    preprocessing = None
    if method == "RT_shapelet":
        stump_hyper_dict = {k: v for k, v in hyper_dict.items() if k.startswith('shapelet.')}
        hyper_dict = {k: v for k, v in hyper_dict.items() if not k.startswith('shapelet.')}
        stump = ShapeletTreeStumpClassifier(**stump_hyper_dict)
        model = RuleTreeClassifier(**hyper_dict, base_stumps=[stump])
    elif method == "pre_RT_shapelet":
        stump_hyper_dict = {k: v for k, v in hyper_dict.items() if k.startswith('shapelet.')}
        hyper_dict = {k: v for k, v in hyper_dict.items() if not k.startswith('shapelet.')}
        preprocessing = Shapelets(**stump_hyper_dict)
        model = RuleTreeClassifier(**hyper_dict, base_stumps=[DecisionTreeStumpClassifier()])
    elif method == 'GenTS':
        raise NotImplementedError("GenTS is not yet implemented")
    elif method == 'Hydra':
        model = HydraClassifier(**hyper_dict)
    elif method == 'MultiRocketHydraClassifier':
        model = MultiRocketHydraClassifier(**hyper_dict)
    elif method == 'RocketClassifier':
        model = RocketClassifier(**hyper_dict)
    elif method == 'MiniRocketClassifier':
        model = MiniRocketClassifier(**hyper_dict)
    elif method == 'MultiRocketClassifier':
        model = MultiRocketClassifier(**hyper_dict)
    elif method == 'ResNet':
        model = ResNet(**hyper_dict)
    elif method == 'MLPClassifier':
        model = MLPClassifier(**hyper_dict)
    elif method == 'InceptionTimeClassifier':
        model = InceptionTimeClassifier(**hyper_dict)
    elif method == 'BOSSEnsemble':
        model = BOSSEnsemble(**hyper_dict)
    elif method == 'MrSEQLClassifier':
        model = MrSEQLClassifier(**hyper_dict)
    elif method == 'MrSQMClassifier':
        model = MrSQMClassifier(**hyper_dict)
    elif method == 'MUSE':
        model = MUSE(**hyper_dict)
    elif method == 'WEASEL':
        model = WEASEL(**hyper_dict)
    elif method == 'WEASELV2':
        model = WEASEL_V2(**hyper_dict)
    elif method == 'KNeighborsTimeSeriesClassifier':
        model = KNeighborsTimeSeriesClassifier(**hyper_dict)
    elif method == 'ProximityForest':
        model = ProximityForest(**hyper_dict)
    elif method == 'ProximityTree':
        model = ProximityTree(**hyper_dict)
    elif method == 'Catch22Classifier_RF':
        model = Catch22Classifier(**hyper_dict)
    elif method == 'Catch22Classifier_DT':
        model = Catch22Classifier(**hyper_dict)
    elif method == 'HIVECOTEV1':
        model = HIVECOTEV1(**hyper_dict)
    elif method == 'HIVECOTEV2':
        model = HIVECOTEV2(**hyper_dict)
    elif method == 'CanonicalIntervalForestClassifier':
        model = CanonicalIntervalForestClassifier(**hyper_dict)
    elif method == 'DrCIFClassifier':
        model = DrCIFClassifier(**hyper_dict)
    elif method == 'IntervalForestClassifier':
        model = IntervalForestClassifier(**hyper_dict)
    elif method == 'BORF':
        model = RidgeClassifier(**hyper_dict['RidgeCV'])
        del hyper_dict['RidgeCV']
        preprocessing = BORF(**hyper_dict)

    return model, preprocessing

def run(method, hyper_dict, hyper_stump_dict, dataset_name, df: pd.DataFrame):
    hyper_dict = copy(hyper_dict)

    filename_hash = '|'.join([
        dataset_name,
        method,
        hashlib.md5('|'.join([f'{x}' for x in hyper_dict.values()]).encode()).hexdigest(),
        hashlib.md5('|'.join([f'{x}' for x in hyper_stump_dict.values()]).encode()).hexdigest(),
    ]) + '.csv'
    if os.path.exists(f"res_tmp/{dataset_name}/{filename_hash}"):
        return pd.read_csv(f"res_tmp/{dataset_name}/{filename_hash}")

    scores = copy(hyper_dict) | {'method': method}
    scores |= copy(hyper_stump_dict)

    y = df[df.columns[-1]].values
    X = df[df.columns[:-1]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scores_kv = []
    skf = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    hyper_dict_original = copy(hyper_dict)
    for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        hyper_dict = copy(hyper_dict_original)
        X_train_kf, X_val, y_train_kf, y_val = (X_train[train_index], X_train[test_index],
                                                y_train[train_index], y_train[test_index])
        model, preprocessing = _get_model(method, hyper_dict)

        start = time.time()
        try:
            if preprocessing is not None:
                X_train_kf = preprocessing.fit(X_train).transform(X_train_kf)
                X_val = preprocessing.transform(X_val)
                if method == 'BORF':
                    X_train_kf = np.arcsinh(X_train_kf)
                    X_val = np.arcsinh(X_val)
            model.fit(X_train_kf, y_train_kf)
        except Exception as e:
            traceback.print_exc()
            return pd.DataFrame()
        end = time.time()

        y_pred_val = model.predict(X_val)

        dict_res = {f'{k}_val': v for k, v in evaluate_clf(y_val, y_pred_val).items()}
        scores_kv.append(dict_res)

    hyper_dict = hyper_dict_original.copy()

    model, preprocessing = _get_model(method, hyper_dict)

    try:
        process = psutil.Process(os.getpid())
        before = process.memory_info().rss
        start = time.time()
        if preprocessing is not None:
            X_train = preprocessing.fit_transform(X_train)
            X_test = preprocessing.transform(X_test)
            if method == 'BORF':
                X_train = np.arcsinh(X_train)
                X_test = np.arcsinh(X_test)
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
            | {f'{k}_test': v for k, v in evaluate_clf(y_test, y_pred, y_pred_proba=None).items()}
    )

    df_val = pd.DataFrame(scores_kv).mean().to_frame().T

    df = pd.DataFrame.from_dict([scores])
    df[df_val.columns] = df_val
    try:
        df.to_csv(f"res_tmp/{dataset_name}/{filename_hash}", index=False)
        pickle.dump(model, open(f"res_tmp/{dataset_name}/{filename_hash.replace('.csv', '.pickle')}", "wb"))
    except Exception as e:
        raise e

    return df


def main():
    datasets = dict() #small_datasets
    for dataset_fun in small_datasets + medium_datasets + big_datasets:
        dataset_name, df = dataset_fun()
        print(dataset_name, df.shape)
        for scaler_name, scaler_fun in scalers.items():
            if scaler_fun is None:
                datasets[dataset_name] = df
            else:
                df_scaled = pd.DataFrame(scaler_fun().fit_transform(df.iloc[:, :-1]).reshape(df.shape[0], -1),
                                         columns=df.columns[:-1])
                df_scaled[df.columns[-1]] = df.iloc[:, -1]
                datasets[f'{dataset_name}_{scaler_name}'] = df_scaled


    n_process = int(psutil.cpu_count(logical=True) / n_jobs)+2
    print(n_process)

    processes = []
    dataframes = []
    single_t = False
    with ProcessPoolExecutor(max_workers=n_process) as executor:
        for dataset_name, df in datasets.items():
            if dataset_name in ['wine', 'ionosphere']:
                continue
            if not os.path.exists("res_tmp/"):
                os.mkdir("res_tmp/")
            if not os.path.exists("res_tmp/" + dataset_name):
                os.mkdir("res_tmp/" + dataset_name)

            if single_t:
                for model, hyper_dict, hyper_stump_dict in tqdm(list(get_hyperparameters(df)),
                                                                desc=f"{dataset_name}: submitting experiments"):
                    processes.append(run(model, hyper_dict, hyper_stump_dict, dataset_name, df))

                for process in tqdm(processes, desc=f"{dataset_name}: collecting results"):
                    dataframes.append(process)

            else:
                for model, hyper_dict, hyper_stump_dict in tqdm(list(get_hyperparameters(df)),
                                                                desc=f"{dataset_name}: submitting experiments"):
                    processes.append((dataset_name,
                                      executor.submit(run, model, hyper_dict, hyper_stump_dict, dataset_name, df)))

                #for process in tqdm(processes, desc=f"{dataset_name}: collecting results"):
                #    try:
                #        res = process.result()
                #        dataframes.append(res)
                #    except Exception as e:
                #        traceback.print_exc()


        it = tqdm(processes)
        for dataset_name, process in it:
            try:
                it.set_description(f"{dataset_name}: collecting results")
                res = process.result()
                dataframes.append(res)
            except Exception as e:
                traceback.print_exc()

if __name__ == '__main__':
    main()