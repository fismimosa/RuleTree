from numba import jit , njit
import random
import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from Competitors.algorithmic_fairness.efficient_algorithms.clustering import load_dataset, calc_balance, calc_fairness_error, calc_clustering_objective, update_centroids, update_centroids_median, clustering, sort_and_valuation, find_distances, find_distances_fast, calc_distance_a, calc_distance, find_k_initial_centroid, k_random_index, dual_print


def run_EFA_init(datasets: list):
    n0 = 100  # number of p=0 points in metric space
    V = n0  # Threshold for p=0
    K = [5]  # No of clusters
    A = 5  # No of attributes
    iterations = 1  # maximum iteration in clustering
    runs = 1
    option = 'Kmeans'  # Kmedian
    run_EFA(dataset, K, iterations)


def run_EFA(df, K_choices, iterations, runs=1, option='Kmeans'):
    # Assuming `A` is the number of attributes (columns without the label)
    #global A
    #A = df.shape[1] -1 # Number of features (excluding the label)
    A = 5

    ct = ColumnTransformer([
        ('std_scaler', StandardScaler(), make_column_selector(dtype_include=[np.number])),
        ("cat", OrdinalEncoder(), make_column_selector(dtype_include="object"))],
        remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=os.cpu_count())

    #df = ct.fit_transform(df)
    print(df)
    df = df.iloc[1:100]
    # Initialize the labels list to store cluster labels
    labels = [-1] * len(df)

    for K in K_choices:
        print("K ==", K)

        # Initialize for multiple runs if needed
        for run in range(runs):
            np.random.seed(run)
            random.seed(run)
            #print(f"Run: {run}")

            # Initialize clustering
            k_centroid = find_k_initial_centroid(df, K, A)
            hashmap_points = {tuple(row[:-1]): 0 for index, row in df.iterrows()}
            #print("hashmap_points run EFA", hashmap_points)

            for iteration in range(iterations):
                #print("iteration", iteration)
                # Find distances and sort based on valuation
                dist = find_distances(k_centroid, df, A)
                #print("after dist", dist)
                sorted_valuation = sort_and_valuation(dist, A)
                #print("after sorted", sorted_valuation)

                # Perform clustering and update labels
                cluster_assign, labels = clustering(df, sorted_valuation, hashmap_points, K, labels)
                print("labels", labels)
                #print("cluster assign", cluster_assign)
                #print("after cluster_assign")

                # Update centroids
                if option == 'Kmeans':
                    k_centroid = update_centroids(cluster_assign, K, A)
                    #print("Final centroids:", k_centroid)
                else:
                    k_centroid = update_centroids_median(cluster_assign, K)

            # Optionally, print out details about the clustering
            #print(f"Labels after run {run}: {labels}")

    return labels


if __name__ == "__main__":
    dataset = load_dataset(
        r"C:\Users\fedev\Downloads\ParTree-main\ParTree-main\Competitors\algorithmic_fairness\Datasets\adult_p.csv")
    run_EFA_init(dataset)