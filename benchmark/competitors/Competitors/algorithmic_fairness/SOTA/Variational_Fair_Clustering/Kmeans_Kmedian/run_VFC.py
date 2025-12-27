import itertools
import os.path
import time
import argparse
from collections import Counter
import warnings
from time import sleep
import os.path as osp
from multiprocessing import Manager
import numpy as np
import pandas as pd
import pkg_resources
import psutil as psutil
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from tqdm.auto import tqdm
#from test_fair_clustering import fair_clustering
from Competitors.algorithmic_fairness.SOTA.Variational_Fair_Clustering.Kmeans_Kmedian.src.fair_clustering import fair_clustering
from Competitors.algorithmic_fairness.SOTA.Variational_Fair_Clustering.Kmeans_Kmedian.src.dataset_load import read_dataset, dataset_names
from Competitors.algorithmic_fairness.SOTA.Variational_Fair_Clustering.Kmeans_Kmedian.src.utils import setup_shared_variables
#ParTree-main.Competitors.algorithmic fairness.SOTA.Variational Fair Clustering (Ziko).Kmeans_Kmedian.src.fair_clustering
from Competitors.algorithmic_fairness.SOTA.Variational_Fair_Clustering.Kmeans_Kmedian.src.utils import setup_shared_variables
import multiprocessing

import ParTree.algorithms.measures_utils as measures

def run_VFC_init():
    parser = argparse.ArgumentParser(description="Run Variational Fair Clustering")
    parser.add_argument('--data_dir', type=str,
                        default=r'C:\Users\fedev\Downloads\ParTree-main\ParTree-main\Experiments\datasets\real\adult',
                        help='Directory where the dataset is located')
    parser.add_argument('--dataset', type=str, default='Adult', help='Name of the dataset')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--K', type=int, default=5, help='Number of clusters')
    # Add other arguments as needed

    args = parser.parse_args()

    # Pass the parsed arguments to your main function
    run_VFC(args, args.data_dir,
            r'C:\Users\fedev\Downloads\ParTree-main\ParTree-main\Experiments\prova')


def run_VFC(args, dataset: str, res_folder):
    has_y = "_y.zip" in dataset
    print("dentro")

    #df = pd.read_csv(dataset, index_col=None)
    y = None
    if has_y:
        y = df[df.columns[-1]]
        df = df.drop(columns=[df.columns[-1]])

    #data_dir = r'C:\Users\fedev\Downloads\ParTree-main\ParTree-main\Experiments\datasets\real\adult\adult.csv'
    manager = Manager()
    SHARED_VARS = manager.dict()
    seedValue = args.seed
    #SHARED_VARS = setup_shared_variables()
    #process1 = multiprocessing.Process(target=other_function, args=(SHARED_VARS,))
    #process1.start()
    #process1.join()
    dataset = args.dataset
    #runNo = args.runNo
    kArg = args.K
    #cluster_option = args.cluster_option
    #data_dir = osp.join(args.data_dir, dataset)
    #data_dir = osp.join(args.data_dir, 'adult_p.csv')
    data_dir = args.data_dir
    #output_path = osp.join(data_dir, dataset + "_K_" + str(kArg))  # +"_Run_"+str(runNo))

    #X_org, demograph, K = read_dataset("Adult", data_dir, args)

    ######################

    # Corrected data directory path
    #data_dir = os.path.join(args.data_dir, 'adult')  # Assuming 'adult' is the correct directory for the dataset

    # Ensure the directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    savepath_compare = os.path.join(data_dir, f"{dataset}_{kArg}_{seedValue}.npz")

    if not os.path.exists(savepath_compare):
        X_org, demograph, K = read_dataset(dataset, data_dir, args)
        if X_org.shape[0] > 200000:
            np.savez_compressed(savepath_compare, X_org=X_org, demograph=demograph, K=K)
        else:
            np.savez(savepath_compare, X_org=X_org, demograph=demograph, K=K)
    else:
        datas = np.load(savepath_compare, allow_pickle=True)
        X_org = datas['X_org']
        demograph = datas['demograph']
        K = datas['K'].item()  # Convert numpy object to scalar as needed
        datas.close()  # Close the file after loading the data

    #############################

    print("X_org", X_org)
    print("demograph", demograph)


    # Compute V_list and u_V
    unique_demograph = np.unique(demograph)
    V_list = [np.array(demograph == j) for j in unique_demograph]
    N = len(demograph)
    V_sum = [v.sum() for v in V_list]
    u_V = [x / N for x in V_sum]

    hyperparams_name = ["K", "u_V", "V_list", "lmbda", "L", "fairness", "method", "C_init"]

    parameters = [
        [2, 3, 5, 10, 20],  # K
        [u_V], #u_V
        [V_list],  # V_list
        [0.0, 0.2, 0.5, 1],  # lmbda
        [0, 1, 2, 5],  # L
        [True], #fairness
        ['kmeans'], #method
        ['kmeans_plus'] #C_init
    ]

    els_bar = tqdm(list(itertools.product(*parameters)), position=2, leave=False)
    for els in els_bar:
        try:
            els_bar.set_description("_".join([str(x) for x in els]) + ".csv")

            colNames = hyperparams_name + ["time", "silhouette", "calinski_harabasz", "davies_bouldin"]
            if has_y:
                colNames += ["r_score", "adj_rand", "mut_info_score", "adj_mutual_info_score", "norm_mutual_info_score",
                             "homog_score", "complete_score", "v_msr_score", "fwlks_mallows_score"]

            filename = "VFC-" \
                       + dataset.split("/")[-1].split("\\")[-1] + "-" \
                       + ("_".join([str(x) for x in els]) + ".csv")

            if os.path.exists(res_folder + filename):
                continue

            ct = ColumnTransformer([
                ('std_scaler', StandardScaler(), make_column_selector(dtype_include=['int', 'float'])),
                ("cat", OrdinalEncoder(), make_column_selector(dtype_include="object"))],
                remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=os.cpu_count())

            #X = ct.fit_transform(df)
            X = X_org
            #_, _, X = prepare_data(X, els[-2], els[-1])
            #print("SHARED VARS PRIMA FAIR CLUSTERING", SHARED_VARS, type(SHARED_VARS))

            C, l, elapsed, S, E = fair_clustering(SHARED_VARS, X, 2, u_V, V_list, 0, 0, True, 'kmeans', 'kmeans_plus',0)
            #print("l", l)
            # Creating a dictionary from l with indices as keys
            l_count = Counter(l)
            #print("Count of unique values in l:", l_count)

            start = time.time()
            #cpt.fit(X)
            stop = time.time()

            #row = list(els) + [stop - start] + measures.get_metrics_uns(X, l)
            #if has_y:
            #    row += measures.get_metrics_s(l, y)

            #pd.DataFrame([row], columns=colNames).to_csv(res_folder + filename, index=False)
        except Exception as e:
            print(f"Errore dataset {dataset}, parametri {'_'.join([str(x) for x in els]) + '.csv'}")
            raise e


#import os



if __name__ == '__main__':
    run_VFC_init()

    #args = argparse.Namespace(
    #    data_dir=r'C:\Users\fedev\Downloads\ParTree-main\ParTree-main\Experiments\datasets\real\adult',
    #    dataset='Adult',
    #    seed=0,
        # Add other necessary default values or parameters here
    #    K=5,  # Example of setting 'K' directly if your functions need it
        #seed=42  # Another example parameter
    #)
    #dataset_path = osp.join(args.data_dir, args.dataset)
    #print("Attempting to read from:", dataset_path)
    #df = pd.read_csv(dataset_path, index_col=None)

    #run_VFC(args, args.data_dir,
    #        r'C:\Users\fedev\Downloads\ParTree-main\ParTree-main\Experiments\prova')
