from numba import jit , njit
import time
import numpy as np
import pandas as pd
import random
        
def k_random_index(df,K):
        # return k random indexes in range of dataframe
        return random.sample(range(0, len(df)), K)
    
def find_k_initial_centroid(df,K,A):
        centroids = []    # make of form [ [x1,y1]....]
    
        rnd_idx = k_random_index(df,K)
        for i in rnd_idx:
            coordinates =[]
            for a in range(0,A):
                coordinates.append(df.loc[i][a])
            centroids.append(coordinates)   #df is X,Y,....., Type
    
        return centroids
    
    #nOt using
def calc_distance(x1, y1, x2, y2):
        # returns the euclidean distance between two points
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    
def calc_distance_a(centroid, point):
        #print('çalculating distance\n')
    
        sum_ = 0
    
        for i in range(0, len(centroid)):
            sum_ = sum_ + (centroid[i]-point[i])**2
    
        return sum_ #**0.5
        
@njit(parallel=False)
def find_distances_fast(k_centroids, df, A):
        
        dist = np.zeros((len(k_centroids),len(df),A+2),np.float64)
        Kcnt = 0 
        for c in k_centroids:  #K-centroid is of form [ c1=[x1,y1.....z1], c2=[x2,y2....z2].....]
           
            l = np.zeros((len(df),A+2),np.float64)
            
          
            index = 0 
            for row in df:                # row is now x,y,z......type
                # append all coordinates to point
                dis = np.sum((c- row[:A])**2)#calc_distance_a(c, point)
                #Processing the vector for list
                row_list = np.array([dis])
                #append distance or l norm
                row_list = np.append(row_list,row[:A+1])
                #append all coordinates #append type of this row
      
                l[index] = row_list
                index = index + 1
                #l.append([calc_distance(c[0], c[1], row[0], row[1]), row[0], row[1], row[2]])  # [dist, X, Y,....Z , type]
                # l contains list of type [dist,X,Y.....,Z,type] for each points in metric space
            dist[Kcnt]= l
            Kcnt = Kcnt + 1
    
        # return dist which contains distances of all points from every centroid
    
        return dist
    
def sort_and_valuation(dist, A):
        sorted_val = []
        for each_centroid_list in dist:
            each_centroid_list_sorted = sorted(each_centroid_list, key=lambda x: (x[A+1], x[0]))  # A+1 is index of type , 0 is dist
            sorted_val.append(each_centroid_list_sorted)
    
            # sort on basis of type & then dist.
            # Now all whites are towards start and all black are after white as they have additional V added to their valuation
            # Among the whites, the most closest is at start of list as it has more valuation.
            # Similarly sort the black points among them based on distance as did with white
    
        return sorted_val
    
def clustering(sorted_valuation, hashmap_points,K):
        n = len(hashmap_points.keys())  # total number of points in metric space
       
        cluster_assign = []
    
        for i in range(0, K):
            cluster_assign.append([])  # initially all clusters are empty
        
        map_index_cluster = []
        for i in range(0,K+2):
            map_index_cluster.append(0)
            #initially check all sorted evaluation from 0th index 
        
        number_of_point_alloc = 0
        curr_cluster = 0
        
        # until all points are allocated
        while number_of_point_alloc != n:  # As convergence is guaranteed that all points will be allocated to some cluster set
            start_inde = map_index_cluster[curr_cluster % K]
            
            for inde in range(start_inde,len(sorted_valuation[curr_cluster % K])):
                each = sorted_valuation[curr_cluster % K][inde]
                # each is (dist,X,Y,....Z,type)
            
               
                if hashmap_points[tuple(each[1: -1])] == 0:    # each is (dist, X,Y,....Z, type)
                    cluster_assign[curr_cluster].append(each)
                    hashmap_points[tuple(each[1: -1])] = 1
                    number_of_point_alloc += 1
                    map_index_cluster[curr_cluster % K] = inde  #next time start from here as isse prev all allocated
                    break
    
            curr_cluster = (curr_cluster + 1) % K
    
        return cluster_assign
    
def calc_clustering_objective(k_centroid, cluster_assign,K):
        cost = 0
    
        for k in range(0, K):
    
            for each in cluster_assign[k]:  #each is (dist, X,Y,....,Z,type)
                dd = calc_distance_a(k_centroid[k], each[1:-1])
                cost = cost + (dd)
    
        return cost
    
def update_centroids(cluster_assign,K,A):
    
        new_centroids = []
        for k in range(0, K):
    
            sum_a = []
    
            for i in range(0, A):
                sum_a.append(0)
    
            for each in cluster_assign[k]:
                sum_a = [sum(x) for x in zip(sum_a, each[1:-1])]
                #each is (dist,X,Y,.....Z,type)
           # print('sum a is '+str(sum_a))
            new_coordinates = []
            for a in range(0, A):
                new_coordinates.append(sum_a[a] / len(cluster_assign[k]))
            new_centroids.append(new_coordinates)
            k=k+1
    
    
    
        return new_centroids
    
    
from collections import Counter
    
def calc_balance(cluster_assign, K):
        S_k = []  # balance of each k cluster
        
        for k in range(K):
            class_counts = Counter()  # Dictionary to store counts of each class
            total_count = 0
            
            for each in cluster_assign[k]:
                class_label = int(each[-1])  # Extract class label (assuming last element is the label)
                class_counts[class_label] += 1
                total_count += 1
            
            if len(class_counts) > 1:  # Ensure we have at least two classes in the cluster
                min_ratios = [class_counts[i] / class_counts[j] for i in class_counts for j in class_counts if i != j]
                S_k.append(min(min_ratios))
            else:
                S_k.append(0)
        
        return min(S_k)


class FRAC_model:
    """Fairness-aware Clustering (FRAC) Model"""

    def __init__(self, K, iterations=300, option='Kmeans', seed_value=42, verbose=False):
        """Initialize the FRAC model with a random seed and result storage."""
        self.seed = seed_value
        self.K = K
        self.iterations = iterations
        self.verbose = verbose
        self.option = option

        self.list_obj_K = []        # Stores objective costs over different K values
        self.list_balance_K = []    # Stores balance measures over different K values
        self.list_centroids_K = []  # Stores centroids for different K values

        self.k_centroid = None
        self.A = None

    def assign_to_nearest_centroid(self, X, k_centroid, A):
        """Assign each row in df_with_target to the nearest centroid."""
        closest_centroids = []  # List to store closest centroid index for each row
        df_ref = pd.DataFrame(X)
        
        for index, row in df_ref.iloc[:, :A].iterrows():  # Consider only first A columns
            min_distance = float('inf')  # Initialize with large value
            closest_idx = -1  # Initialize closest centroid index
            
            for i, center in enumerate(k_centroid):
                distance = calc_distance_a(center[:A], row.values)  # Compute distance to centroid
                
                if distance < min_distance:
                    min_distance = distance
                    closest_idx = i  # Update with closest centroid index
            
            closest_centroids.append(closest_idx)  # Store closest centroid index
        
        return np.array(closest_centroids)
        

    def predict(self, X):
        """Predicts the X labels

        Args:
            X as numpy array
            
        """
        return self.assign_to_nearest_centroid(X, self.list_centroids_K, self.A)

    
    def fit(self, X, ):
        """Fits the model using the given parameters.

        Args:
            X (numpy array): Does contain the protected attribute but does not contain the target class/y
            K (int): Number of clusters.
            protected_attr_list (list):    protected attribute.
            verbose (bool): Whether to print debug information.
            iterations (int): Number of iterations for clustering.
            option (str): Clustering algorithm to use ('Kmeans' or other).
        """

        # Set random seed
        np.random.seed(self.seed)
        random.seed(self.seed)

        df = pd.DataFrame(X)
        #df['prot_attr'] = self.protected_attr_list


        A = df.shape[1] - 1  # Number of considered features excluding the protected attribute+
        self.A = A
        
        if self.verbose:
            print(f'Considering {A} features (excluding the protected attribute).')

        # Initialize result lists
        list_obj_K = []
        list_balance_K = []
        list_centroids_K = []

        if self.verbose:
            print(f'Running clustering with K={self.K}, option={self.option}, iterations={self.iterations}')

        # Your clustering logic goes here

        list_obj_iter = []
        list_balance_iter = []
        list_centroids_iter = []

        K = [self.K]
        kk = K[0]

        list_obj_run = []
        list_balance_run = []
        list_centroids_run = []
        
        k_centroid = find_k_initial_centroid(df, kk, A)
        cluster_assignment = [[] for _ in range(K[0])]

        sum_time = 0
        curr_itr = 0
        prev_objective_cost = -1
        objective_cost = 0
        
        while curr_itr <= self.iterations:
            start = time.process_time()
    
            df1 = df.values
            k_centroids1 = np.array(k_centroid)
            
            dist = find_distances_fast(k_centroids1, df1,A)

            valuation = sort_and_valuation(dist, A)
            
            hash_map = {tuple(row[1:]): 0 for _, row in df.iterrows()}
         
            cluster_assignment = clustering(valuation, hash_map, K[0])

            balance = calc_balance(cluster_assignment, K[0])
            clustering_cost = calc_clustering_objective(k_centroid, cluster_assignment, K[0])

            if curr_itr != 0:
                prev_objective_cost = objective_cost
                
            objective_cost = np.round(clustering_cost, 3)
            
            list_balance_iter.append(str(balance))
            list_obj_iter.append(str(objective_cost))
            list_centroids_iter.append(k_centroid.copy())  # Store a copy to avoid mutation issues


            if abs(objective_cost - prev_objective_cost) <= 30:
                break
            
            if self.option == 'Kmeans':
                k_centroid = update_centroids(cluster_assignment, K[0],A)
            
            stop = time.process_time()
            sum_time += (stop - start)

            curr_itr += 1

        balance_converged = calc_balance(cluster_assignment, K[0])
        clustering_cost_converged = calc_clustering_objective(k_centroid, cluster_assignment, K[0])
        
        list_obj_run.append(clustering_cost_converged)
        list_balance_run.append(balance_converged)
        list_centroids_run.append(k_centroid.copy())  # Append a copy to avoid reference issues

        list_obj_K.append(np.mean(np.array(list_obj_run)))
        list_balance_K.append(np.mean(np.array(list_balance_run)))
        list_centroids_K.append(list_centroids_run)  # Store all centroid updates for K
    
       
        self.list_obj_K = list_obj_K     # Stores objective costs over different K values
        self.list_balance_K = list_balance_K # Stores balance measures over different K values
        self.list_centroids_K = list_centroids_K[0][0] # Stores centroids for different K values
