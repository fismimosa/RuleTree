import numpy as np
import random
from sklearn.cluster import KMeans
import math
import utils
from scipy.optimize import linprog
import copy


def k_means_mod(X, k):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(X)
    return kmeans.cluster_centers_, kmeans.score(X), pred_y


# Trivial Fair Clustering:
def trivial_clustering(point_ls, new_pat_ls, new_pat_dic, k, gamma):
    # Kmeans clustering
    X = np.array(point_ls)
    centers, score, labels = k_means_mod(X, k)

    # centers are needed for the assignment problem
    centers = list(centers)

    # coeffcients of the objective
    min_dist = 0.0
    min_cen = centers[0]
    for cen in centers:
        point1 = np.array(cen)
        dis_cen = 0
        for i in range(0, len(new_pat_ls)):
            ids, vecs = new_pat_ls[i]
            point2 = np.array(new_pat_dic[ids])
            dist = np.linalg.norm(point1 - point2)
            dis_cen = dis_cen + (dist * dist)
        if min_dist == 0:  # first time
            min_dist = dis_cen
            min_cen = cen
        else:
            if dis_cen < min_dist:
                min_dist = dis_cen
                min_cen = cen

    return min_dist


def construct_radius_dict(new_pat_ls, neighbours, num):
    point_radius_dict = dict()
    for i in range(0, len(new_pat_ls)):
        ids, vecs = new_pat_ls[i]
        nbr_list = neighbours[ids]
        nbr_id, nbr_dis = nbr_list[num]
        point_radius_dict[ids] = nbr_dis
    return point_radius_dict


def compute_dist_nbr(point_set):
    neighbours = dict()
    for p in point_set:
        nbr_list = []
        for x in point_set:
            point1 = np.array(point_set[p])
            point2 = np.array(point_set[x])
            # calculating Euclidean distance
            # using linalg.norm()
            dist = np.linalg.norm(point1 - point2)
            nbr_list.append((x, dist))

        # sort the neighbor list and compute n_i
        nbr_list.sort(key=lambda x: x[1])
        neighbours[p] = nbr_list

    return neighbours


def choose_new_center(new_pt_set, nbrs, num):
    '''choose the point with minium radius'''
    final_id = -1
    final_dis = -1
    # print(len(new_pt_set))
    for i in range(0, len(new_pt_set)):
        ids, vecs = new_pt_set[i]
        nbr_list = nbrs[ids]
        nbr_id, nbr_dis = nbr_list[num]
        if final_dis == -1:
            final_dis = nbr_dis
            final_id = nbr_id
        else:
            if final_dis > nbr_dis:
                final_dis = nbr_dis
                final_id = nbr_id
    return final_id


# fair evaluation
def compute_individual_fair_dict(point_set, gamma, clusters, m_v_list, k, is_norm):
    if is_norm:
        ratio = k / len(clusters.keys())
    else:
        ratio = 1
    gamma_dict = dict()
    for c in clusters:
        ids, _ = clusters[c]
        for i in ids:
            match = 0.0
            for j in ids:
                if i != j:
                    if utils.gamma_match(point_set[i], point_set[j], gamma):
                        match += 1
            gamma_dict[i] = match

    fair_dict = dict()
    unsatisfied_fair = 0  # this computes how many points needed to make this fair
    unsatisfied_fair_fraction = 0  # this computes what fraction of points needed to make this fair
    unfair_count = 0
    for point in gamma_dict:
        diff = gamma_dict[point] - m_v_list[point] * ratio
        if diff >= 0:
            fair_dict[point] = ["fair", diff]
        else:
            frac = gamma_dict[point] / (1.0 * m_v_list[point] * ratio)
            fair_dict[point] = ["unfair", diff]
            unsatisfied_fair_fraction += frac
            unsatisfied_fair += (-1.0 * diff)
            unfair_count += 1
    if unfair_count == 0:
        unsatisfied_fair_fr = 0
    else:
        unsatisfied_fair_fr = unsatisfied_fair_fraction / unfair_count

    return gamma_dict, fair_dict, unsatisfied_fair_fr, unsatisfied_fair


def compute_alpha(gamma_dict, m_v_list, ratio):
    alpha = []
    for point in gamma_dict:
        if m_v_list[point] == 0:
            new_al = 0
        else:
            new_al = gamma_dict[point] / (m_v_list[point] * ratio)
        alpha.append(new_al)
    return alpha


def compute_fair_cluster(fair_dict, clusters):
    # one can also compute the difference
    percent_ls = []
    for c in clusters:
        ids, _ = clusters[c]
        score = 0.0
        for i in ids:
            if fair_dict[i][0] == "fair":
                score += 1
        percent_ls.append(score / len(ids))

    return min(percent_ls), max(percent_ls), sum(percent_ls) / len(percent_ls), fair_dict


def fair_evaluate(gamma_dict, fair_dict, clusters, m_v_list, k, is_norm):
    # alpha computation
    if is_norm:
        ratio = k / len(clusters.keys())
    else:
        ratio = 1
    alpha = compute_alpha(gamma_dict, m_v_list, ratio)
    mean_alpha = sum(alpha) / len(alpha)
    above_k = [i for i in alpha if i > (1.0 / k)]
    above_k2 = [i for i in alpha if i > (2.0 / k)]
    above_k3 = [i for i in alpha if i > (3.0 / k)]

    frac_min, frac_max, frac_avg, fair_dict = compute_fair_cluster(fair_dict, clusters)

    count_fair = 0
    for i in fair_dict:
        if fair_dict[i][0] == "fair":
            count_fair += 1
    pt_ind_fair = count_fair / len(list(fair_dict.values()))
    return mean_alpha, min(alpha), len(above_k) / len(fair_dict), len(above_k2) / len(fair_dict), len(above_k3) / len(fair_dict), frac_min, frac_max, frac_avg, pt_ind_fair


def distance_score(Clusters, point_set):
    score = 0.0
    for c in Clusters:
        ids, cen = Clusters[c]
        if cen is None:
            # cen = point_set[list(point_set.keys())[c]]
            cen = np.mean([point_set[id_] for id_ in ids], axis=0)
        for i in ids:
            point1 = np.array(point_set[i])
            point2 = np.array(cen[:len(point1)])
            dist = np.linalg.norm(point1 - point2)
            score = score + (dist * dist)

    return score


def get_clusters(label_dict, center_dict):
    clusters = dict()
    for i in label_dict:
        if label_dict[i] not in clusters:
            clusters[label_dict[i]] = [i]
        else:
            clusters[label_dict[i]].append(i)
    for i in clusters:
        x = clusters[i]
        y = center_dict[i]
        clusters[i] = (x, y)
    return clusters


def fairness_given_clusters(k, gamma, m_v_list, eval_pat_dic, eval_feature_dict, clusters):
    keys = list(eval_pat_dic.keys())
    unique_clusters = np.unique(clusters)
    new_clusters = {c: ([], None) for c in unique_clusters}
    for i, key in enumerate(keys):
        new_clusters[clusters[i]][0].append(key)

    gamma_dict, fair_dict, unsatis_frac, unsatis_num = compute_individual_fair_dict(eval_feature_dict, gamma, new_clusters, m_v_list, k, is_norm=True)
    mean_alpha, min_alpha, k_above, k_above2, k_above3, frac_min, frac_max, frac_avg_norm, pt_ind_fair_norm = fair_evaluate(gamma_dict, fair_dict, new_clusters, m_v_list, k, is_norm=True)

    gamma_dict, fair_dict, unsatis_frac, unsatis_num = compute_individual_fair_dict(eval_feature_dict, gamma, new_clusters, m_v_list, k, is_norm=False)
    mean_alpha, min_alpha, k_above, k_above2, k_above3, frac_min, frac_max, frac_avg, pt_ind_fair = fair_evaluate(gamma_dict, fair_dict, new_clusters, m_v_list, k, is_norm=False)

    dis_score = distance_score(new_clusters, eval_pat_dic)
    cluster_sizes = [len(new_clusters[c][0]) for c in new_clusters]

    return unsatis_frac, unsatis_num, mean_alpha, min_alpha, k_above, k_above2, k_above3, frac_min, frac_max, frac_avg, pt_ind_fair, dis_score, len(new_clusters.keys()), frac_avg_norm, pt_ind_fair_norm, cluster_sizes


def get_LP_clusters(new_pat_ls, opt_var, centers):
    # cumulative probability array
    counter = 0
    prob_array = [0] * len(opt_var)
    for i in range(0, len(new_pat_ls)):
        cen_c = 0
        sums = 0
        for j in range(0, len(centers)):
            ind = counter * len(centers) + cen_c
            sums += opt_var[ind]
            prob_array[ind] = sums
            cen_c += 1
        counter += 1

    # cluster assignment
    label_dict = dict()
    counter = 0
    for i in range(0, len(new_pat_ls)):
        pointid, vec = new_pat_ls[i]
        cen_c = 0
        sums = 0
        prob = random.random()
        for j in range(0, len(centers)):
            ind = counter * len(centers) + cen_c
            if prob < prob_array[ind]:
                if j not in label_dict:
                    label_dict[j] = [pointid]
                else:
                    label_dict[j].append(pointid)
                break
            else:
                if j + 1 == len(centers):
                    if j not in label_dict:
                        label_dict[j] = [pointid]
                    else:
                        label_dict[j].append(pointid)

                else:
                    cen_c += 1

        counter += 1

    # clusters
    clusters = dict()

    for i in label_dict:
        x = label_dict[i]
        y = centers[i]
        clusters[i] = (x, y)

    return clusters


def LP_fair(point_ls, new_pat_ls, new_pat_dic, k, gamma, m_v_list, matched_pts, eval_pat_dict, eval_feature_dict):
    # Kmeans clustering
    X = np.array(point_ls)
    centers, score, labels = k_means_mod(X, k)

    # centers are needed for the assignment problem
    centers = list(centers)

    # coeffcients of the objective
    obj_coeff_ls = []
    for i in range(0, len(new_pat_ls)):
        ids, vecs = new_pat_ls[i]
        point1 = np.array(new_pat_dic[ids])
        for cen in centers:
            point2 = np.array(cen)
            dist = np.linalg.norm(point1 - point2)
            obj_coeff_ls.append(dist * dist)

    # variable values bounded
    bnd = [(0, 1)] * (len(new_pat_ls) * len(centers))

    # equality that all sum up to 1
    rhs_eq = [1] * len(new_pat_ls)

    lhs_eq = []
    # number of equation
    for i in range(0, len(new_pat_ls)):

        # within an equation - go over all the variables and set coeffcients

        lhs_eq_it = []
        for j in range(0, len(new_pat_ls)):
            if j == i:
                for curr_cen in centers:
                    lhs_eq_it.append(1)
            else:
                for curr_cen in centers:
                    lhs_eq_it.append(0)
        lhs_eq.append(lhs_eq_it)

        # based on the gamma match
    lhs_ineq = []
    rhs_ineq = [0] * (len(new_pat_ls) * len(centers))

    # number of equations
    for i in range(0, len(new_pat_ls)):
        for cen in centers:

            # within an equation - go over all the variables and set coeffcients
            # print(cen)
            lhs_ineq_it = []
            for j in range(0, len(new_pat_ls)):
                for curr_cen in centers:
                    # print(curr_cen)
                    if curr_cen.all() != cen.all():
                        lhs_ineq_it.append(0)
                    else:
                        # now equation
                        ids_i, vecs_i = new_pat_ls[i]
                        ids, vecs = new_pat_ls[j]
                        if j == i:
                            lhs_ineq_it.append(m_v_list[ids])
                        else:
                            if ids in matched_pts[ids_i]:
                                lhs_ineq_it.append(-1)
                            else:
                                lhs_ineq_it.append(0)

            lhs_ineq.append(lhs_ineq_it)

    # print("computing the solution for", run_algo, "times: ",ii)

    opt = linprog(c=obj_coeff_ls, A_ub=lhs_ineq, b_ub=rhs_ineq, A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd, method="revised simplex")
    print(opt.fun)
    print(opt.message)
    opt_var = opt.x

    # check
    counter = 0
    sums_ls = []
    for i in range(0, len(new_pat_ls)):
        cen_c = 0
        sums = 0
        for cen in centers:
            ind = counter * len(centers) + cen_c
            sums += opt_var[ind]
            cen_c += 1
        counter += 1
        sums_ls.append(sums)

    # Cluster construction
    Clusters = get_LP_clusters(new_pat_ls, opt_var, centers)

    # print statistics about clusters
    for c in Clusters:
        ids, cen = Clusters[c]
        print("Label: ", c, " its length: ", len(ids))

    # fair evaluation
    gamma_dict, fair_dict, unsatis_frac, unsatis_num = compute_individual_fair_dict(eval_feature_dict, gamma, Clusters, m_v_list, k, is_norm=True)
    mean_alpha, min_alpha, k_above, k_above2, k_above3, frac_min, frac_max, frac_avg_norm, pt_ind_fair_norm = fair_evaluate(gamma_dict, fair_dict, Clusters, m_v_list, k, is_norm=True)

    gamma_dict, fair_dict, unsatis_frac, unsatis_num = compute_individual_fair_dict(eval_feature_dict, gamma, Clusters, m_v_list, k, is_norm=False)
    mean_alpha, min_alpha, k_above, k_above2, k_above3, frac_min, frac_max, frac_avg, pt_ind_fair = fair_evaluate(gamma_dict, fair_dict, Clusters, m_v_list, k, is_norm=False)

    dis_score = distance_score(Clusters, eval_pat_dict)
    cluster_sizes = [len(Clusters[c][0]) for c in Clusters]

    return opt.x, opt.fun, unsatis_frac, unsatis_num, mean_alpha, min_alpha, k_above, k_above2, k_above3, frac_min, frac_max, frac_avg, pt_ind_fair, dis_score, len(Clusters.keys()), frac_avg_norm, pt_ind_fair_norm, cluster_sizes
