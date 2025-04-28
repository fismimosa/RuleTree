import math
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import traceback
import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split

from RuleTree.stumps.classification import DecisionTreeStumpClassifier, ObliqueDecisionTreeStumpClassifier, \
    PivotTreeStumpClassifier, MultiplePivotTreeStumpClassifier, ObliquePivotTreeStumpClassifier
from RuleTree.stumps.classification.MultipleObliquePivotTreeStumpClassifier import \
    MultipleObliquePivotTreeStumpClassifier
from RuleTree.stumps.classification.PartialPivotTreeStumpClassifier import PartialPivotTreeStumpClassifier
from RuleTree.stumps.classification.PartialProximityTreeStumpClassifier import PartialProximityTreeStumpClassifier
from benchmark.Hybrid.HybridHyper import get_hyperparameters, n_jobs
from benchmark.Hybrid.HybridReaders import all_datasets, small_datasets, medium_datasets, big_datasets

n = 16

os.environ["OMP_NUM_THREADS"] = f"{n}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{n}"
os.environ["MKL_NUM_THREADS"] = f"{n}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{n}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{n}"

import hashlib
import os.path
import time
from concurrent.futures import ProcessPoolExecutor
from copy import copy

import pandas as pd
import psutil
from tqdm.auto import tqdm

from RuleTree import RuleTreeClassifier
from benchmark.evaluation_utils import evaluate_expl, evaluate_clf

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def compute_measures(y_test, y_pred, model):
    measures = evaluate_clf(y_test, y_pred, y_pred_proba=None)
    measures |= evaluate_expl(model)

    return measures


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

    scores = copy(hyper_dict)
    scores |= copy(hyper_stump_dict)

    y = df[df.columns[-1]].values
    X = df[df.columns[:-1]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    if method == "RT":
        base_stumps = []
        if 'base_stumps' not in hyper_dict:
            print()
        for stump in hyper_dict["base_stumps"]:
            if stump == "PartialPivotTreeStumpClassifier":
                base_stumps.append(PartialPivotTreeStumpClassifier(**hyper_stump_dict[stump]))
            elif stump == "PartialProximityTreeStumpClassifier":
                base_stumps.append(PartialProximityTreeStumpClassifier(**hyper_stump_dict[stump]))
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
        model = RuleTreeClassifier(**hyper_dict, base_stumps=base_stumps)
    elif method == "DT":
        model = DecisionTreeStumpClassifier(**hyper_dict)
    else:
        raise ValueError(f"Unknown base_method: {method}")

    start = time.time()
    try:
        model.fit(X_train, y_train)
    except:
        return pd.DataFrame()
    end = time.time()

    if method in ['DT', 'RT']:
        start_pred = time.time()
        y_pred = model.predict(X_test)
        end_pred = time.time()
        scores |= ({'train_time': end - start, 'pred_time': end_pred - start_pred}
                   | compute_measures(y_test, y_pred, model)
                   )
    else:
        raise ValueError(f"Unknown model: {model}")

    df = pd.DataFrame.from_dict([scores])
    try:
        df.to_csv(f"res_tmp/{dataset_name}/{filename_hash}", index=False)
    except Exception as e:
        raise e

    return df


def main():
    datasets = dict([x() for x in small_datasets + medium_datasets + big_datasets])
    n_process = math.ceil(psutil.cpu_count(logical=True) / n_jobs)
    print(n_process)

    n_process = 6

    with ProcessPoolExecutor(max_workers=n_process) as executor:
        for dataset_name, df in datasets.items():
            if not os.path.exists("res_tmp/"):
                os.mkdir("res_tmp/")
            if not os.path.exists("res_tmp/" + dataset_name):
                os.mkdir("res_tmp/" + dataset_name)

            processes = []
            dataframes = []

            single_t = False

            if single_t:
                for model, hyper_dict, hyper_stump_dict in tqdm(list(get_hyperparameters(df)),
                                                                desc=f"{dataset_name}: submitting experiments"):
                    processes.append(run(model, hyper_dict, hyper_stump_dict, dataset_name, df))

                for process in tqdm(processes, desc=f"{dataset_name}: collecting results"):
                    dataframes.append(process)

            else:

                for model, hyper_dict, hyper_stump_dict in tqdm(list(get_hyperparameters(df)),
                                                                desc=f"{dataset_name}: submitting experiments"):
                    processes.append(executor.submit(run, model, hyper_dict, hyper_stump_dict, dataset_name, df))

                for process in tqdm(processes, desc=f"{dataset_name}: collecting results"):
                    try:
                        res = process.result()
                        dataframes.append(res)
                    except Exception as e:
                        traceback.print_exc()

            pd.concat(dataframes, ignore_index=True).to_csv(f"hybrid_benchmark_{dataset_name}.csv", index=False)


if __name__ == '__main__':
    main()
