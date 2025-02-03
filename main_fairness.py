from concurrent.futures.process import ProcessPoolExecutor

import pandas as pd
import numpy as np
import psutil
from progress_table import ProgressTable

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

from RuleTree import RuleTreeCluster
from RuleTree.stumps.regression.FairTreeStumpRegressor import FairTreeStumpRegressor
from RuleTree.tree.RuleTree import RuleTree


def fair_entropy(labels:np.ndarray, prot_attr=np.ndarray):
    classes = np.unique(prot_attr)
    labels_list = np.unique(labels)

    entropy=0

    for clu_id in labels_list:
        cls, count = np.unique(prot_attr[labels == clu_id], return_counts=True)
        min_cls = cls[np.argmin(count)]

        Na = np.sum(labels == clu_id)
        Nab = np.sum((labels==clu_id) & (prot_attr==min_cls))

        entropy += Nab/Na * np.log2(Nab/Na)

    return -entropy

def balance(labels:np.ndarray, prot_attr=np.ndarray):
    res = []

    for pr_attr in np.unique(prot_attr):
        r = np.sum(prot_attr == pr_attr)/len(labels)
        for cl_id in np.unique(labels):
            ra = np.sum((labels == cl_id) & (prot_attr == pr_attr))/np.sum(labels == cl_id)
            rab= r/ra if ra != 0 else 0
            rab_1 = 1/rab if rab != 0 else 1
            res.append(min(rab, rab_1))


    return min(res)


def main():
    df = pd.read_csv('datasets/CLU/adult.csv')[["age", "fnlwgt", "education-num", "sex",
                                                "capital-gain", "capital-loss", "hours-per-week"]].head(500)

    min_perc = 1.
    for s in df.sex.unique():
        print(s, len(df[df.sex == s])/len(df))
        min_perc = min(min_perc, len(df[df.sex == s])/len(df))

    df["sex"] = LabelEncoder().fit_transform(df.sex)

    kmeans = KMeans(n_clusters=2, random_state=42)#
    kmeans.fit(df.values)

    RT = RuleTreeCluster(max_depth=1, bic_eps=.2)
    fairRT = RuleTreeCluster(max_depth=1, bic_eps=.2, base_stumps=FairTreeStumpRegressor(sensible_attribute=3,
                                                                             k_anonymity=.26,
                                                                             l_diversity=1,
                                                                             t_closeness=.5,
                                                                             strict=True
                                                                             ))
    RT.fit(df.values)
    fairRT.fit(df.values)
    RuleTree.print_rules(RT.get_rules())
    RuleTree.print_rules(fairRT.get_rules())

    print("sil sk", silhouette_score(X=df.values, labels=kmeans.labels_), sep='\t')
    print("sil RT", silhouette_score(X=df.values, labels=RT.predict(df.values)), sep='\t')
    print("sil fRT", silhouette_score(X=df.values, labels=fairRT.predict(df.values)), sep='\t')

    print("bal dt", balance(labels=np.zeros((len(df), )), prot_attr=df.sex), sep='\t')
    print("bal sk", balance(labels=kmeans.labels_, prot_attr=df.sex), sep='\t')
    print("bal RT", balance(labels=RT.predict(df.values), prot_attr=df.sex), sep='\t')
    print("bal fRT", balance(labels=fairRT.predict(df.values), prot_attr=df.sex), sep='\t')


def run(df, X, k, l, t, s):
    fairRT = RuleTreeCluster(max_depth=1, bic_eps=.2, base_stumps=FairTreeStumpRegressor(sensible_attribute=3,
                                                                                     k_anonymity=k,
                                                                                     l_diversity=l,
                                                                                     t_closeness=t,
                                                                                     strict=s
                                                                                     ))
    fairRT.fit(X)

    return silhouette_score(X, fairRT.predict(X)), balance(labels=fairRT.predict(X), prot_attr=df.sex)

def main_multi():
    df = pd.read_csv('datasets/CLU/adult.csv')[["age", "fnlwgt", "education-num", "sex",
                                                "capital-gain", "capital-loss", "hours-per-week"]].head(500)

    df["sex"] = LabelEncoder().fit_transform(df.sex)

    min_perc = 1.
    for s in df.sex.unique():
        print(s, len(df[df.sex == s]) / len(df))
        min_perc = min(min_perc, len(df[df.sex == s]) / len(df))

    k_values = np.arange(0, min_perc, .02)
    X=df.values
    l=1
    t_values=np.arange(0, 1., .05)
    s=True

    results = []
    with ProcessPoolExecutor(max_workers=psutil.cpu_count()) as pool:
        for t in tqdm(t_values, position=0, leave=True):
            for k in tqdm(k_values, position=1, leave=False):
                results.append(pool.submit(run, X, k, l, t, s))

    res = []
    for t in tqdm(t_values, position=0, leave=True):
        for k, result in zip(tqdm(k_values, position=1, leave=False), results):
            result = result.result()
            res.append((k, l, t, s, result[0], result[1]))
            print(f"k={k}: Silhouette Score={result[0]}, Balance={result[1]}")

    pd.DataFrame(res, columns=['k', 'l', 't', 's', 'sil', 'bal']).to_csv("res_fair.csv", index=False)

if __name__ == '__main__':
    main()