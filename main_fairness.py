from concurrent.futures.process import ProcessPoolExecutor

import pandas as pd
import numpy as np
import psutil
from progress_table import ProgressTable

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, fowlkes_mallows_score
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

from RuleTree import RuleTreeCluster
from RuleTree.stumps.regression.FairTreeStumpRegressor import FairTreeStumpRegressor
from RuleTree.tree.RuleTree import RuleTree
from RuleTree.utils.fairness_metrics import balance_metric, max_fairness_cost, privacy_metric

def compute_measures(X, clu_id, target, prot_attr, ideal):
    return [round(x, 3) for x in [
        silhouette_score(X, clu_id, metric='euclidean'),
        fowlkes_mallows_score(target, clu_id),
        balance_metric(clu_id, prot_attr),
        max_fairness_cost(clu_id, prot_attr, ideal),
    ]]


def main():
    df = pd.read_csv('datasets/CLF/titanic.csv').drop(columns=['PassengerId', 'Embarked', 'Cabin_letter'])
    df = df[~df.Cabin_n.isin(['T', 'D'])]

    df["Sex"] = LabelEncoder().fit_transform(df.Sex)

    min_perc = 1.
    mfc_map = dict()
    for s in df.Sex.unique():
        print(s, len(df[df.Sex == s])/len(df), sep='\t')
        mfc_map[s] = len(df[df.Sex == s])*1./len(df)
        min_perc = min(min_perc, len(df[df.Sex == s])/len(df))

    print(mfc_map)

    X = df.drop(columns=["Survived"]).values.astype(float)
    target = df["Survived"].values
    prot_attr = df["Sex"].values

    depth = 4
    n_jobs=1 #Usare tutti e 512 rallenta

    kmeans = KMeans(n_clusters=2**depth, random_state=42)#

    RT = RuleTreeCluster(max_depth=2, bic_eps=.2)
    fairRT_privacy = RuleTreeCluster(max_depth=depth, bic_eps=.2, base_stumps=FairTreeStumpRegressor(sensible_attribute=1,
                                                                                                     penalty="privacy",
                                                                                                     k_anonymity=.9*min_perc,
                                                                                                     l_diversity=1,
                                                                                                     t_closeness=.0,
                                                                                                     strict=True,
                                                                                                     n_jobs=n_jobs
                                                                                                 ))

    fairRT_privacy_no_t = RuleTreeCluster(max_depth=depth, bic_eps=.2, base_stumps=FairTreeStumpRegressor(sensible_attribute=1,
                                                                                                          penalty="privacy",
                                                                                                          k_anonymity=.9*min_perc,
                                                                                                          l_diversity=1,
                                                                                                          t_closeness=.0,
                                                                                                          strict=True,
                                                                                                          use_t=False,
                                                                                                          n_jobs=n_jobs,
                                                                                                 ))

    fairRT_balance = RuleTreeCluster(max_depth=depth, bic_eps=.2, base_stumps=FairTreeStumpRegressor(sensible_attribute=1,
                                                                                                     penalty="balance",
                                                                                                     n_jobs=n_jobs
                                                                                                 ))


    fairRT_mfc = RuleTreeCluster(max_depth=depth, bic_eps=.2, base_stumps=FairTreeStumpRegressor(sensible_attribute=1,
                                                                                                 penalty="mfc",
                                                                                                 ideal_distribution=mfc_map,
                                                                                                 n_jobs=n_jobs
                                                                                                 ))
    print(*["\t", "sil", "fms", "bal", "mfc"], sep='\t\t')
    kmeans.fit(X)
    print("sk-kmeans", *compute_measures(X, kmeans.labels_, target, prot_attr, mfc_map), sep='\t\t')

    RT.fit(X)
    print("RuleTree", *compute_measures(X, RT.predict(X), target, prot_attr, mfc_map), sep='\t\t')

    fairRT_privacy.fit(X)
    print("fRT privacy", *compute_measures(X, fairRT_privacy.predict(X), target, prot_attr, mfc_map), sep='\t\t')

    fairRT_privacy_no_t.fit(X)
    print("fRT pri_no_t", *compute_measures(X, fairRT_privacy_no_t.predict(X), target, prot_attr, mfc_map), sep='\t\t')

    fairRT_balance.fit(X)
    print("fRT balance", *compute_measures(X, fairRT_balance.predict(X), target, prot_attr, mfc_map), sep='\t\t')

    fairRT_mfc.fit(X)
    print("fRT mfc eq_dt", *compute_measures(X, fairRT_mfc.predict(X), target, prot_attr, mfc_map), sep='\t\t')


if __name__ == '__main__':
    main()