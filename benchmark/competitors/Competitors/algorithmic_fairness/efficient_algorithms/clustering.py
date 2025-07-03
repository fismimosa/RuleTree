from numba import jit , njit
import random
import numpy as np
import pandas as pd
import os


def load_Bank(data_dir=''):
    data_dir = data_dir
    _path = 'bank-full_p_6col.csv'
    data_path = os.path.join(data_dir, _path)

    K = 10

    df = pandas.read_csv(data_path, sep=',')
    # print(df.head())
    # print(len(df))

    return df


def load_Adult(data_dir=''):
    data_dir = data_dir
    _path = 'adult_p.csv'
    data_path = os.path.join(data_dir, _path)

    K = 10

    df = pandas.read_csv(data_path, sep=',')
    # print(df.head())
    # print(len(df))

    return df


def dual_print(f, *args, **kwargs):
    # print(*args,**kwargs)
    print(*args, **kwargs, file=f)


def load_dataset(csv_name):
    # read the dataset from csv_name and return as pandas dataframe
    df = pd.read_csv(csv_name, header=None)
    return df


def k_random_index(df, K):
    # return k random indexes in range of dataframe
    return random.sample(range(0, len(df)), K)


def find_k_initial_centroid(df, K, A):
    centroids = []  # make of form [ [x1,y1]....]

    rnd_idx = k_random_index(df, K)
    for i in rnd_idx:
        coordinates = []
        for a in range(0, A):
            coordinates.append(df.loc[i][a])
        centroids.append(coordinates)  # df is X,Y,....., Type

    return centroids


# nOt using
def calc_distance(x1, y1, x2, y2):
    # returns the euclidean distance between two points
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def calc_distance_a(centroid, point):
    sum_ = 0

    for i in range(0, len(centroid)):
        sum_ = sum_ + (centroid[i] - point[i]) ** 2

    return sum_  # **0.5


@njit(parallel=False)
def find_distances_fast(k_centroids, df):
    dist = np.zeros((len(k_centroids), len(df), A + 2), np.float64)
    Kcnt = 0
    for c in k_centroids:  # K-centroid is of form [ c1=[x1,y1.....z1], c2=[x2,y2....z2].....]

        l = np.zeros((len(df), A + 2), np.float64)

        index = 0
        for row in df:  # row is now x,y,z......type
            # append all coordinates to point
            dis = np.sum((c - row[:A]) ** 2)  # calc_distance_a(c, point)
            # Processing the vector for list
            row_list = np.array([dis])
            # append distance or l norm
            row_list = np.append(row_list, row[:A + 1])
            # append all coordinates #append type of this row

            l[index] = row_list
            index = index + 1
            # l.append([calc_distance(c[0], c[1], row[0], row[1]), row[0], row[1], row[2]])  # [dist, X, Y,....Z , type]
            # l contains list of type [dist,X,Y.....,Z,type] for each points in metric space
        dist[Kcnt] = l
        Kcnt = Kcnt + 1

    # return dist which contains distances of all points from every centroid

    return dist


def find_distances(k_centroids, df, A):
    dist = []
    for c in k_centroids:  # K-centroid is of form [ c1=[x1,y1.....z1], c2=[x2,y2....z2].....]
        #c = int(c)
        c = [float(num) for num in c]
        l = []
        #print("c", c)

        for index, row in df.iterrows():  # row is now x,y,z......type
            point = []
            for a in range(0, A):
                a = int(a)
                #print("a inside find distances", a)
                #a = [float(num) for num in a]
                point.append(row.iloc[a])  # append all coordinates
                #print("point inside find dist", point)

            point = [float(num) for num in point]
            #print("point", point)
            dis = calc_distance_a(c, point)
            # Processing the vector for list
            row_list = [dis]
            #print("row_list", row_list)
            # append distance or l norm

            for a in range(0, A):
                row_list.append(row.iloc[a])  # append all coordinates
                #print("row_list", row.iloc[a])

            print("a", a)
            row_list.append(row.iloc[a + 1])  # append type of this row
            print("row", row)

            l.append(row_list)
            #print("row_list", row_list)
            # l.append([calc_distance(c[0], c[1], row[0], row[1]), row[0], row[1], row[2]])  # [dist, X, Y,....Z , type]
            # l contains list of type [dist,X,Y.....,Z,type] for each points in metric space
        dist.append(l)

    # return dist which contains distances of all points from every centroid

    return dist


def sort_and_valuation(dist, A):
    sorted_val = []
    for each_centroid_list in dist:
        each_centroid_list_sorted = sorted(each_centroid_list,
                                           key=lambda x: (x[A + 1], x[0]))  # A+1 is index of type , 0 is dist
        sorted_val.append(each_centroid_list_sorted)

        # sort on basis of type & then dist.
        # Now all whites are towards start and all black are after white as they have additional V added to their valuation
        # Among the whites, the most closest is at start of list as it has more valuation.
        # Similarly sort the black points among them based on distance as did with white

    return sorted_val


def clustering(df, sorted_valuation, hashmap_points, K, labels):
    #print("df", df)
    n = len(hashmap_points.keys())  # total number of points in metric space
    #print("Total points:", n)
    cluster_assign = []
    for i in range(0, K):
        cluster_assign.append([])  # initially all clusters are empty

    map_index_cluster = []
    for i in range(0, K + 2):
        map_index_cluster.append(0)  # Keep track of the index for each cluster
    number_of_point_alloc = 0
    curr_cluster = 0

    while number_of_point_alloc != n:
        print(f"Loop Start: curr_cluster={curr_cluster}, number_of_point_alloc={number_of_point_alloc}, start_index={map_index_cluster[curr_cluster % K]}")
        start_index = map_index_cluster[curr_cluster % K]

        #no_progress = True

        # Loop through possible points for the current cluster
        for index in range(start_index, len(sorted_valuation[curr_cluster % K])):
            each = sorted_valuation[curr_cluster % K][index]
            #print("each", each[1:-1])
            #point_tuple = tuple(each[1:-1])

            #print("Checking point:", point_tuple, "with hashmap_points[point_tuple]:", hashmap_points[point_tuple])

            if hashmap_points[tuple(each[1:-1])] == 0:  # Check if this point has been assigned
                #print(f"Assigning point {point_tuple} to cluster {curr_cluster}")
                cluster_assign[curr_cluster].append(each)
                hashmap_points[tuple(each[1:-1])] = 1  # Mark this point as assigned
                number_of_point_alloc += 1
                # Find the original index of this point in the dataframe
                original_index = np.where((df.iloc[:, :-1].values == np.array(tuple(each[1:-1]))).all(axis=1))[0][0]
                labels[original_index] = curr_cluster  # Update the label of this point

                #number_of_point_alloc += 1
                map_index_cluster[curr_cluster % K] = index  # Move to the next point for this cluster
                #no_progress = False
                break
            #else:
                # Skip this point and move to the next
            #    map_index_cluster[curr_cluster % K] += 1

        #if no_progress:
            #print(f"No progress in current loop: curr_cluster={curr_cluster}")
            # Directly move to the next cluster if all points are assigned or not suitable
           # map_index_cluster[curr_cluster % K] = len(sorted_valuation[curr_cluster % K])

        curr_cluster = (curr_cluster + 1) % K

    return cluster_assign, labels



def update_centroids_median(cluster_assign, K):
    new_centroids = []
    for k in range(0, K):

        cAk = np.array(cluster_assign[k])
        cAk = np.delete(cAk, [0, -1], axis=1)
        if len(cAk) % 2 == 0 and len(cAk) > 0:
            cc = [np.median(np.array(cAk[:-1])[:, cl]) for cl in range(0, cAk.shape[1])]
            new_centroids.append(cc)
        elif len(cAk) % 2 != 0 and len(cAk) > 0:
            cc = [np.median(np.array(cAk)[:, cl]) for cl in range(0, cAk.shape[1])]
            new_centroids.append(cc)
        elif len(cAk) == 0:
            print("Error: No centroid found updation error")

    return new_centroids


def update_centroids(cluster_assign, K, A):
    new_centroids = []
    print("K", K)

    for k in range(K):  # Simplified range(0, K) to range(K)
        print("k in for", k)

        sum_a = [0] * A  # Directly initialize sum_a with zeros

        # Debug: Check the size of cluster_assign[k]
        print(f"Size of cluster_assign[{k}]:", len(cluster_assign[k]))

        # Summing up all the coordinates for each point in the cluster
        for each in cluster_assign[k]:
            # Debug: Print current point being processed
            #print(f"Processing point: {each}")

            # Update sum_a by adding the corresponding coordinates
            sum_a = [sum_a[i] + float(each[i + 1]) for i in range(A)]

        #print("Sum of coordinates:", sum_a)

        # If there are no points in this cluster
        if len(cluster_assign[k]) == 0:
            #print(f"No points in cluster {k}, assigning zero centroid.")
            new_centroids.append([0] * A)
        else:
            # Calculate new centroid by averaging each coordinate
            new_coordinates = [sum_a[a] / len(cluster_assign[k]) for a in range(A)]
            new_centroids.append(new_coordinates)

        #print("New centroid for cluster", k, ":", new_centroids[-1])

    return new_centroids


def calc_clustering_objective(k_centroid, cluster_assign, K):
    cost = 0

    for k in range(0, K):

        for each in cluster_assign[k]:  # each is (dist, X,Y,....,Z,type)
            dd = calc_distance_a(k_centroid[k], each[1:-1])
            cost = cost + (dd)

    return cost


def calc_fairness_error(df, cluster_assign, K):
    U = []  # distribution of each type in original target dataset for each J = 0 , 1....
    P_k_sum_over_j = []  # distribution in kth cluster  sum_k( sum_j(   Uj * j wale/total_in_cluster ) )

    f_error = 0
    cnt_j_0 = 0
    cnt_j_1 = 0
    #  cnt_j_2 = 0   Uncomment in case of Bank (Ternanry Datatset)
    cnt = 0
    for index, row in df.iterrows():
        if row.iloc[-1] == 1:
            cnt_j_1 += 1
        elif row.iloc[-1] == 0:
            cnt_j_0 += 1
        #  elif row.iloc[-1] == 2:
        #    cnt_j_2 += 1

        cnt += 1

    U.append(cnt_j_0 / cnt)
    U.append(cnt_j_1 / cnt)
    # U.append(cnt_j_2 / cnt)

    for k in range(0, K):  # for each cluster

        for j in range(0, len(U)):  # for each demographic group

            cnt_j_cluster = 0
            cnt_total = 0

            for each in cluster_assign[k]:
                if int(each[-1]) == j:  # each is (dist,X, Y.....,Z,type)
                    cnt_j_cluster += 1
                cnt_total += 1

            if cnt_j_cluster != 0 and cnt_total != 0:
                P_k_sum_over_j.append(-U[j] * np.log((cnt_j_cluster / cnt_total) / U[j]))
            else:
                P_k_sum_over_j.append(0)  # log(0)=0 considered

    for each in P_k_sum_over_j:
        f_error += each

    return f_error


def calc_balance(cluster_assign, K):
    S_k = []  # balance of each k cluster
    balance = 0  # min (S_k)

    for k in range(0, K):
        cnt_j_0 = 0
        cnt_j_1 = 0
        # cnt_j_2 = 0   Uncomment in case of ternary dataset Bank
        cnt = 0
        for each in cluster_assign[k]:

            if int(each[-1]) == 1:
                cnt_j_1 += 1
            elif int(each[-1]) == 0:
                cnt_j_0 += 1
            # elif int(each[-1]) == 2:
            #     cnt_j_2 += 1

            cnt += 1

        if cnt_j_0 != 0 and cnt_j_1 != 0:  # and cnt_j_2!= 0:
            S_k.append(min([cnt_j_0 / cnt_j_1,
                            cnt_j_1 / cnt_j_0]))  # , cnt_j_1 / cnt_j_2 , cnt_j_2 / cnt_j_1 , cnt_j_0 / cnt_j_2, cnt_j_2 / cnt_j_0 ]))
        elif cnt_j_0 == 0 or cnt_j_1 == 0:  # or cnt_j_2==0:
            S_k.append(0)

    balance = min(S_k)

    return balance
