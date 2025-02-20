from itertools import product

import pandas as pd
from numba.cuda.cudadrv.nvrtc import nvrtc_program

from benchmark.Fair.FairReaders import read_titanic

n_clus = [2]#[2, 4, 8, 16]
n_jobs = 16

hyper_params_FairStump = [
    {
        "method": ["FairRT_privacy"],
        "sensible_attribute": [0],
        "penalization_weight": [.2, .5, .8],
        "n_jobs": [n_jobs],

        "penalty": ["privacy"],
        "k_anonymity": [.8, .9, .95],  # da moltiplicare per il dataset
        "l_diversity": [.45, .7, 1.],  # da moltiplicare
        "t_closeness": [.0, .1],
        "strict": [True, False],
        "use_t": [True, False],
    },

    {
        "method": ["FairRT_balance"],
        "sensible_attribute": [0],
        "penalization_weight": [.1, .3, .5, .7, .9],
        "n_jobs": [n_jobs],

        "penalty": ["balance"],
    },

    {
        "method": ["FairRT_mfc"],
        "sensible_attribute": [0],
        "penalization_weight": [.2, .5, .8],
        "n_jobs": [n_jobs],

        "penalty": ["mfc"],
        "ideal_distribution": ['equal', 'dataset'], # to fix
    },
]


hyper_params_list = ([

                        {
                            "base_method": ['kmeans'],
                            "n_clusters": n_clus,
                            "random_state": [42],
                        },
                        {
                            "base_method": ['RT'],
                            "bic_eps": [.0, .1, .2],
                            "max_leaf_nodes": n_clus,
                        },
                         {
                             "base_method": ['DB'],
                             "eps": [.1, .25, .5, .75, 1.],
                             "metric": ["euclidean", "cosine", "correlation"]
                         }
                     ] + [
                        {
                            "base_method": ['FRT'],
                            "bic_eps": [.0, .1, .2],
                            "max_leaf_nodes": n_clus,
                        }|x for x in hyper_params_FairStump
                     ]
)


def get_hyperparameters(df: pd.DataFrame):
    protected = df.columns.tolist()[0]
    n_protected = len(df[protected].unique())

    min_perc = float("inf")
    mfc_dataset = dict()
    for s in df[protected].unique():
        #print(s, len(df[df[protected] == s]) / len(df), sep='\t')
        mfc_dataset[s] = len(df[df[protected] == s]) * 1. / len(df)
        min_perc = min(min_perc, len(df[df[protected] == s]) / len(df))

    mfc_equal = {k: 1/len(df[protected].unique()) for k in df[protected].unique()}

    hyper_final = []

    for dizionario in hyper_params_list:
        for hyper_val in product(*dizionario.values()):
            hyper_dict = dict(zip(dizionario.keys(), hyper_val))

            if 'k_anonymity' in hyper_dict:
                hyper_dict['k_anonymity'] *= min_perc
            if 'l_diversity' in hyper_dict:
                hyper_dict['l_diversity'] = round(hyper_dict['l_diversity'] * (n_protected -1))

            if 'ideal_distribution' in hyper_dict:
                if hyper_dict['ideal_distribution'] == 'equal':
                    hyper_dict['ideal_distribution'] = mfc_equal
                elif hyper_dict['ideal_distribution'] == 'dataset':
                    hyper_dict['ideal_distribution'] = mfc_dataset

            hyper_final.append(hyper_dict)

    return hyper_final


if __name__ == "__main__":
    _, df = read_titanic()
    hyper_final = get_hyperparameters(df)

    print(len(hyper_final))


