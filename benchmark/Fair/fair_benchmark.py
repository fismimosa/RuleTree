import os

from benchmark.competitors.FRAC_model import FRAC_model
from benchmark.competitors.VFC_model import VFC_model
from benchmark.competitors.kmeanstree import KMeansTree

n = 16

os.environ["OMP_NUM_THREADS"] = f"{n}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{n}"
os.environ["MKL_NUM_THREADS"] = f"{n}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{n}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{n}" # export VECLIB_MAXIMUM_THREADS=4

import hashlib
import os.path
import time
from concurrent.futures import ProcessPoolExecutor
from copy import copy

import pandas as pd
import psutil
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

from RuleTree import RuleTreeCluster
from RuleTree.stumps.regression.FairTreeStumpRegressor import FairTreeStumpRegressor
from RuleTree.utils.fairness_metrics import balance_metric, max_fairness_cost
from benchmark.Fair.FairHyper import get_hyperparameters, n_jobs
from benchmark.Fair.FairReaders import read_titanic, read_bank_marital, read_bank_housing, read_bank_default, \
    read_bank_education, read_bank_age, read_taiwan_credit_marriage, read_taiwan_credit_age, read_taiwan_credit_sex, \
    read_taiwan_credit_education, read_diabetes_race, read_diabetes_age, read_diabetes_gender, read_compass_race, \
    read_compass_sex, read_compass_age
from benchmark.evaluation_utils import evaluate_clu_sup, evaluate_clu_unsup, evaluate_expl

datasets = dict([
    read_titanic(),

    read_bank_marital(),
    read_bank_housing(),
    read_bank_default(),
    read_bank_education(),
    read_bank_age(),

    read_taiwan_credit_education(),
    read_taiwan_credit_marriage(),
    read_taiwan_credit_age(),
    read_taiwan_credit_sex(),

    read_diabetes_age(),
    read_diabetes_gender(),
    read_diabetes_race(),

    read_compass_race(),
    read_compass_sex(),
    read_compass_age(),
])


def compute_measures(X, clu_id, target, prot_attr, ideal, model):
    measures = {
        'balance_metric': balance_metric(clu_id, prot_attr),
        'max_fairness_cost': max_fairness_cost(clu_id, prot_attr, ideal),
    }

    measures |= evaluate_clu_sup(clu_id, target)
    measures |= evaluate_clu_unsup(clu_id, X, X)
    measures |= evaluate_expl(model)

    return measures



def run(hyper, dataset_name, df:pd.DataFrame):
    filename = dataset_name+'|'+'|'.join([f'{x}' for x in hyper.values()])+'.csv'
    filename_hash = dataset_name+'|'+hashlib.md5('|'.join([f'{x}' for x in hyper.values()]).encode()).hexdigest()+'.csv'
    if os.path.exists("res_tmp/"+filename):
        return pd.read_csv("res_tmp/"+filename)
    if os.path.exists(filename_hash):
        return pd.read_csv("res_tmp/"+filename_hash)

    scores = copy(hyper)

    target = df[df.columns[-1]]
    prot_attr = df[df.columns[0]]
    X = df[df.columns[:-1]].values

    mfc_dataset = dict()
    for s in prot_attr.unique():
        mfc_dataset[s] = len(df[prot_attr == s]) * 1. / len(df)

    base_method = hyper["base_method"]
    del hyper["base_method"]

    if base_method == "RT":
        model = RuleTreeCluster(**hyper)
    elif base_method == "kmeans":
        model = KMeans(**hyper)
    elif base_method == "DB":
        model = DBSCAN(n_jobs=n_jobs, algorithm='ball_tree', **hyper)
    elif base_method == "FRT":
        bic_eps = hyper["bic_eps"]
        del hyper["bic_eps"]
        max_leaf_nodes = hyper["max_leaf_nodes"]
        del hyper["max_leaf_nodes"]

        del hyper["method"]

        stump = FairTreeStumpRegressor(**hyper)
        model = RuleTreeCluster(bic_eps=bic_eps, max_leaf_nodes=max_leaf_nodes, base_stumps=stump)
    elif base_method == "KMT":
        model = KMeansTree(n_jobs=n_jobs, **hyper)
    elif base_method == "FRAC":
        model = FRAC_model(**hyper)
    elif base_method == "VFC":
        model = VFC_model(**hyper)

    start = time.time()
    model.fit(X)
    end = time.time()

    if base_method in ['DB', 'kmeans']:
        scores |= {'time': end - start} | compute_measures(X, model.labels_, target, prot_attr, mfc_dataset, model)
    elif base_method == 'KMeansTree':
        scores |= {'time': end - start} | compute_measures(X, model.predict(X), target, prot_attr, mfc_dataset, model.dt)
        scores |= {'accuracy': model.accuracy_}
    else:
        scores |= {'time': end-start} | compute_measures(X, model.predict(X), target, prot_attr, mfc_dataset, model)
    scores["dataset"] = dataset_name

    df = pd.DataFrame.from_dict([scores])
    try:
        df.to_csv("res_tmp/"+filename, index=False)
    except Exception as e:
        df.to_csv("res_tmp/"+filename_hash, index=False)

    return df


def main():
    max_vals = 100

    for dataset_name, df in tqdm(datasets.items(), position=0, leave=False, desc="MinMax"):
        df = pd.DataFrame(MinMaxScaler().fit_transform(df.values), columns=df.columns)
        for col in df.select_dtypes(include=['float64', 'int64']):
            if len(df[col].unique()) > max_vals:
                df[col] = pd.cut(df[col], bins=max_vals, labels=False, duplicates='drop')
        datasets[dataset_name] = df



    for dataset_name, df in datasets.items():
        processes = []
        dataframes = []

        single_t = True

        if single_t:
            for hyper in tqdm(get_hyperparameters(df), desc=f"{dataset_name}: submitting experiments"):
                processes.append(run(hyper, dataset_name, df))

            for process in tqdm(processes, desc=f"{dataset_name}: collecting results"):
                dataframes.append(process)

        else:
            with ProcessPoolExecutor(max_workers=(psutil.cpu_count())) as executor: #//n_jobs
                for hyper in tqdm(get_hyperparameters(df), desc=f"{dataset_name}: submitting experiments"):
                    processes.append(executor.submit(run, hyper, dataset_name, df))

                for process in tqdm(processes, desc=f"{dataset_name}: collecting results"):
                    try:
                        res = process.result()
                        dataframes.append(res)
                    except Exception as e:
                        print(e)


        pd.concat(dataframes, ignore_index=True).to_csv(f"fair_benchmark_{dataset_name}.csv", index=False)



if __name__ == '__main__':
    main()