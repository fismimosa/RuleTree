import math
import random
import numpy as np
import csv


# calculate vector distance and return 1 if distance smaller than gamma.
def gamma_match(point1, point2, gamma):
    assert len(point1) == len(point2)
    return np.exp(-np.linalg.norm(np.array(point1) - np.array(point2))) >= gamma


def choose_randm_points(pat_dict, num):
    #rand_patids = random.sample(pat_dict.keys(), num)
    rand_patids = random.sample(list(pat_dict.keys()), num)
    print("rand_patids", rand_patids)
    new_pat_ls = []
    new_pat_dic = dict()
    print("new_pat_ls", new_pat_ls)
    for i in rand_patids:
        print(i)
        new_pat_ls.append((i, pat_dict[i]))
        new_pat_dic[i] = pat_dict[i]
    point_ls = []
    for i in range(0, len(new_pat_ls)):
        ids, vec = new_pat_ls[i]
        point_ls.append(vec)

    return point_ls, new_pat_ls, new_pat_dic


def compute_thresholds(point_list, features, gamma, threshold, k):
    mvdict = dict()
    gamma_matched = dict()
    for i in point_list:
        match = 0.0
        match_ls = []
        for j in point_list:
            if j != i:
                # if gamma_match(features[i], features[j], gamma) == 1:
                if gamma_match(features[i], features[j], gamma):
                    match += 1
                    match_ls.append(j)
        mvdict[i] = match
        gamma_matched[i] = match_ls

    new_dict = dict()
    for i in mvdict:
        new_dict[i] = threshold * (mvdict[i] / k)
        if len(gamma_matched[i]) < new_dict[i]:
            new_dict[i] = len(gamma_matched[i])
        if new_dict[i] == 0:
            new_dict[i] = 1  # handling 0 matches

    return new_dict, gamma_matched


def cluster_imbalance(cluster_sizes):
    """Compute cluster imbalance from cluster sizes for different runs."""
    std_over_clusters = []
    for i in range(len(cluster_sizes)):
        std_over_clusters.append(np.std(cluster_sizes[i]))
    return np.mean(std_over_clusters), np.std(std_over_clusters)
