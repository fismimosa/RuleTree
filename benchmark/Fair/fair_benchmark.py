import time
from concurrent.futures import ProcessPoolExecutor
from copy import copy

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, fowlkes_mallows_score
from tqdm.auto import tqdm

from RuleTree import RuleTreeCluster
from RuleTree.stumps.regression.FairTreeStumpRegressor import FairTreeStumpRegressor
from RuleTree.utils.fairness_metrics import balance_metric, max_fairness_cost
from benchmark.Fair.FairHyper import get_hyperparameters
from benchmark.Fair.FairReaders import read_titanic, read_bank_marital, read_bank_housing, read_bank_default, \
    read_bank_education, read_bank_age, read_taiwan_credit_marriage, read_taiwan_credit_age, read_taiwan_credit_sex, \
    read_taiwan_credit_education, read_diabetes_race, read_diabetes_age, read_diabetes_gender

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
])


def compute_measures(X, clu_id, target, prot_attr, ideal):
    return {
        'silhouette_score': silhouette_score(X, clu_id, metric='euclidean'),
        'fowlkes_mallows_score': fowlkes_mallows_score(target, clu_id),
        'balance_metric': balance_metric(clu_id, prot_attr),
        'max_fairness_cost': max_fairness_cost(clu_id, prot_attr, ideal),
    }



def run(hyper, dataset_name, df:pd.DataFrame):
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
    else:
        bic_eps = hyper["bic_eps"]
        del hyper["bic_eps"]
        max_depth = hyper["max_depth"]
        del hyper["max_depth"]

        del hyper["method"]

        stump = FairTreeStumpRegressor(**hyper)
        model = RuleTreeCluster(bic_eps=bic_eps, max_depth=max_depth, base_stumps=stump)

    start = time.time()
    model.fit(X)
    end = time.time()

    scores |= {'time': end-start} | compute_measures(X, model.predict(X), target, prot_attr, mfc_dataset)
    scores["dataset"] = dataset_name

    return scores


def main():
    for dataset_name, df in datasets.items():
        processes = []
        dataframes = []

        with ProcessPoolExecutor(max_workers=30*2) as executor:
            for hyper in tqdm(get_hyperparameters(df), desc="submitting experiments"):
                processes.append(executor.submit(run, hyper, dataset_name, df))

            for process in tqdm(processes, desc="collecting results"):
                result = process.result()
                dataframes.append(pd.DataFrame.from_dict([result]))

        pd.concat(dataframes, ignore_index=True).to_csv("fair_benchmark.csv", index=False)



if __name__ == '__main__':
    main()