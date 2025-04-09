import os.path

import pandas as pd
import random
import numpy as np
import utils
import clustering
import data
import sys
import time


def assignment_get(assignments, k, num):
    assignment_prob_keys = assignments[0]
    assignment_probs = np.zeros((num, k))
    for key in assignment_prob_keys:
        assignment_probs[key[1], key[0]] = assignment_prob_keys[key]

    # vals = np.array(list(assignments[0].values())).reshape(k, -1).T
    clusters = []
    for val_probs in assignment_probs:
        if sum(val_probs) == 0:
            clusters.append(random.choices(range(k), k=1)[0])
        else:
            clusters.append(random.choices(range(k), weights=val_probs)[0])
    return clusters


def one_run(pat_dict, feature_dict, k, num, threshold=0.5):
    point_ls, new_pat_ls, new_pat_dic = utils.choose_randm_points(pat_dict, num)
    print("points selected")
    gamma = .2

    # k=10
    print("THRESHOLD------------------", threshold)
    m_v_list, gamma_matched_pts = utils.compute_thresholds(new_pat_dic, feature_dict, gamma, threshold, k)
    print("mv_list computed")

    new_point_ls = []
    new_new_pat_ls = []
    new_new_pat_dic = {}
    for id_, value in new_pat_ls:
        new_value = value + feature_dict[id_]
        new_point_ls.append(new_value)
        new_new_pat_dic[id_] = new_value
        new_new_pat_ls.append((id_, new_value))

    trivial_dis = clustering.trivial_clustering(point_ls, new_pat_ls, new_pat_dic, k, gamma)

    print("*************************** LP **************************************")
    best_fairness = 0
    best_cluster_sizes = None
    best_res = None
    for trial in range(10):
        opt_var, opt_cost, unsatis_frac, unsatis_num, mean_alpha, min_alpha, k_above, k_above2, k_above3, frac_min, frac_max, frac_avg, pt_ind_fair, dis_score, cluster_num, frac_avg_norm, pt_ind_fair_norm, cluster_sizes = \
            clustering.LP_fair(point_ls, new_pat_ls, new_pat_dic, k, gamma, m_v_list, gamma_matched_pts, new_pat_dic, feature_dict)
        dis_norm = dis_score / trivial_dis
        res = [unsatis_frac, unsatis_num, mean_alpha, min_alpha, k_above, k_above2, k_above3, frac_min, frac_max, frac_avg, pt_ind_fair, dis_score, dis_norm, cluster_num, frac_avg_norm, pt_ind_fair_norm]
        if pt_ind_fair > best_fairness:
            best_fairness = pt_ind_fair
            best_res = res
            best_cluster_sizes = cluster_sizes
    return best_res, best_cluster_sizes


def run_all_algos(pat_dict, feature_dict, k, num):
    threshold = 0.5
    performance_dict = {'distance': [], 'number_of_clusters': [], 'fairness_pt_norm': [], 'fairness_avg_norm': [], 'cluster_sizes': []}
    results = []
    for i in range(5):
        print("REPEATING EXPERIMENTS**********************************************************************")
        random.seed(i)
        np.random.seed(i)
        res, cluster_sizes = one_run(pat_dict, feature_dict, k, num, threshold)
        results.append(res)

        performance_dict['distance'].append(res[-4])
        performance_dict['number_of_clusters'].append(res[-3])
        performance_dict['fairness_avg_norm'].append(res[-2])
        performance_dict['fairness_pt_norm'].append(res[-1])
        performance_dict['cluster_sizes'].append(cluster_sizes)

    return performance_dict


if __name__ == '__main__':
    dataset = sys.argv[1]
    k = int(sys.argv[2])
    num = int(sys.argv[3])

    if dataset == 'diabetes':
        feature_set = data.select_feature_set_diabetes()
    elif dataset == 'adult':
        feature_set = data.select_feature_set_adult()
    elif dataset == 'bank':
        feature_set = data.select_feature_set_bank()
    else:
        raise NotImplementedError(f'Dataset: {dataset} is not implemented.')

    pat_dict, feature_dict = data.load_data(dataset_name=dataset, feature_set=feature_set)

    path = f'./results/{dataset}/'
    if not os.path.exists(path + 'performances'):
        os.makedirs(path + 'performances')

    performance_methods = {}
    method = 'LP'
    file_path = path + f"performances/performance_{method}_{k}_{num}.npy"
    if os.path.exists(file_path):
        performance = np.load(file_path, allow_pickle=True).item()
    else:
        performance = run_all_algos(pat_dict, feature_dict, k, num)
        np.save(file_path, performance)
    performance_methods[method] = performance

    with open(path + f"results_{k}_{num}.txt", 'w') as _file:
        for id_ in feature_set:
            _file.write(f"{id_}: {feature_set[id_]}\n")
        _file.write('\n')
        round_point = 3
        round_point2 = 1
        cluster_sizes = performance_methods[method]['cluster_sizes']
        cluster_imbalance_mean, cluster_imbalance_std = utils.cluster_imbalance(cluster_sizes)
        _file.write(
            f"{method:<30} {'Distance: ' + str(np.mean(performance_methods[method]['distance']).round(round_point)) + ' +- ' + str(np.std(performance_methods[method]['distance']).round(round_point)):<30} "
            f"{'Fairness Norm PT: ' + str((np.mean(performance_methods[method]['fairness_pt_norm']).round(round_point) * 100).round(round_point2)) + ' +- ' + str((np.std(performance_methods[method]['fairness_pt_norm']).round(round_point) * 100).round(round_point2)):<40} "
            f"{'Fairness Norm AVG: ' + str((np.mean(performance_methods[method]['fairness_avg_norm']).round(round_point) * 100).round(round_point2)) + ' +- ' + str((np.std(performance_methods[method]['fairness_avg_norm']).round(round_point) * 100).round(round_point2)):<40} "
            f"{'Cluster Num AVG: ' + str(np.mean(performance_methods[method]['number_of_clusters']).round(round_point)) + ' +- ' + str(np.std(performance_methods[method]['number_of_clusters']).round(round_point)):<30} "
            f"{'Cluster Imbalance AVG: ' + str(cluster_imbalance_mean.round(round_point)) + ' +- ' + str(cluster_imbalance_std.round(round_point)):<30}\n")
