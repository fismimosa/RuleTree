import numpy as np
from scipy import sparse
import math
#from sklearn.cluster._kmeans import _k_init
#from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances as ecdist
from sklearn.metrics import pairwise_distances_chunked as pdist_chunk
from sklearn.cluster import KMeans
#from sklearn.cluster.k_means_ import _init_centroids
#from sklearn.cluster._kmeans import _init_centroids
import timeit
import multiprocessing
from numba import  jit
import numexpr as ne

from competitors.Competitors.algorithmic_fairness.SOTA.Variational_Fair_Clustering.Kmeans_Kmedian.src import utils
from competitors.Competitors.algorithmic_fairness.SOTA.Variational_Fair_Clustering.Kmeans_Kmedian.src.bound_update import \
    get_S_discrete, bound_update, normalize_2
from competitors.Competitors.algorithmic_fairness.SOTA.Variational_Fair_Clustering.Kmeans_Kmedian.src.utils import \
    get_fair_accuracy_proportional


#import random
#random.seed(0)
#np.random.seed(0)

# import pdb
def kmeans_update(tmp, SHARED_VARS):
    """
    """
    X = SHARED_VARS['X_s']
    X_tmp = X[tmp, :]
    c1 = X_tmp.mean(axis = 0)

    return c1

#@jit
def reduce_func(D_chunk,start):
    J = np.mean(D_chunk,axis=1)
    return J


def kmedian_update(tmp, SHARED_VARS):
    """

    """
    #X = utils.SHARED_VARS['X_s']
    X = SHARED_VARS['X_s']
    X_tmp = X[tmp,:]
    D = pdist_chunk(X_tmp,reduce_func=reduce_func)
    J = next(D)
    j = np.argmin(J)
    c1 = X_tmp[j,:]
    return c1

def NormalizedCutEnergy(A, S, clustering):
    if isinstance(A, np.ndarray):
        d = np.sum(A, axis=1)

    elif isinstance(A, sparse.csc_matrix):
        
        d = A.sum(axis=1)

    maxclusterid = np.max(clustering)
    nassoc_e = 0;
    num_cluster = 0;
    for k in range(maxclusterid+1):
        S_k = S[:,k]
        if 0 == np.sum(clustering==k):
             continue # skip empty cluster
        num_cluster = num_cluster + 1
        if isinstance(A, np.ndarray):
            nassoc_e = nassoc_e + np.dot( np.dot(np.transpose(S_k),  A) , S_k) / np.dot(np.transpose(d), S_k)
        elif isinstance(A, sparse.csc_matrix):
            nassoc_e = nassoc_e + np.dot(np.transpose(S_k), A.dot(S_k)) / np.dot(np.transpose(d), S_k)
            nassoc_e = nassoc_e[0,0]
    ncut_e = num_cluster - nassoc_e

    return ncut_e

def NormalizedCutEnergy_discrete(A, clustering):
    if isinstance(A, np.ndarray):
        d = np.sum(A, axis=1)

    elif isinstance(A, sparse.csc_matrix):

        d = A.sum(axis=1)

    maxclusterid = np.max(clustering)
    nassoc_e = 0;
    num_cluster = 0;
    for k in range(maxclusterid+1):
        S_k = np.array(clustering == k,dtype=np.float)
        if 0 == np.sum(clustering==k):
             continue # skip empty cluster
        num_cluster = num_cluster + 1
        if isinstance(A, np.ndarray):
            nassoc_e = nassoc_e + np.dot( np.dot(np.transpose(S_k),  A) , S_k) / np.dot(np.transpose(d), S_k)
        elif isinstance(A, sparse.csc_matrix):
            nassoc_e = nassoc_e + np.dot(np.transpose(S_k), A.dot(S_k)) / np.dot(np.transpose(d), S_k)
            nassoc_e = nassoc_e[0,0]
    ncut_e = num_cluster - nassoc_e

    return ncut_e

# @jit
def KernelBound_k(A, d, S_k, N):
    # S_k = S[:,k]
    volume_s_k = np.dot(np.transpose(d), S_k)
    volume_s_k = volume_s_k[0,0]
    temp = np.dot(np.transpose(S_k), A.dot(S_k)) / (volume_s_k * volume_s_k)
    temp = temp * d
    temp2 = temp + np.reshape( - 2 * A.dot(S_k) / volume_s_k, (N,1))

    return temp2.flatten()

#@jit
def km_le(X,M):
    
    """
    Discretize the assignments based on center
    
    """
    e_dist = ecdist(X,M)          
    l = e_dist.argmin(axis=1)
        
    return l

# Fairness term calculation
def fairness_term_V_j(u_j,S,V_j):
    V_j = V_j.astype('float')
    S_term = np.maximum(np.dot(V_j,S),1e-20)
    S_sum = np.maximum(S.sum(0),1e-20)
    S_term = ne.evaluate('u_j*(log(S_sum) - log(S_term))')
    return S_term

def km_discrete_energy(e_dist,l,k):
    tmp = np.asarray(np.where(l== k)).squeeze()
    return np.sum(e_dist[tmp,k])

def compute_energy_fair_clustering(X, C, l, S, u_V, V_list, bound_lambda, A = None, method_cl='kmeans'):
    """
    compute fair clustering energy
    l is cluster no
    """
    J = len(u_V)
    N,K = S.shape
    clustering_E_discrete = []
    if method_cl =='kmeans':
        e_dist = ecdist(X,C,squared =True)
        clustering_E = ne.evaluate('S*e_dist').sum()
        clustering_E_discrete = [km_discrete_energy(e_dist,l,k) for k in range(K)]
        clustering_E_discrete = sum(clustering_E_discrete)

    elif method_cl =='ncut':
        
        clustering_E = NormalizedCutEnergy(A,S,l)
        clustering_E_discrete = NormalizedCutEnergy_discrete(A,l)

    elif method_cl =='kmedian':
        e_dist = ecdist(X,C,squared =True)
        clustering_E = ne.evaluate('S*e_dist').sum()
        clustering_E_discrete = [km_discrete_energy(e_dist,l,k) for k in range(K)]
        clustering_E_discrete = sum(clustering_E_discrete)
    
    # Fairness term 
    fairness_E = [fairness_term_V_j(u_V[j],S,V_list[j]) for j in range(J)]
    fairness_E = (bound_lambda*sum(fairness_E)).sum()
    
    E = clustering_E + fairness_E

    return E, clustering_E, fairness_E, clustering_E_discrete
    
def km_init(X, K,seedValue, C_init, l_init= None):
    
    """
    Initial seeds
    """

    ss = timeit.default_timer()
    x_squared_norms = np.sum(X**2, axis=1)
    random_state = np.random.RandomState(seedValue)
    #print("Type of X:", type(X))
    #print("Shape of X:", X.shape if hasattr(X, 'shape') else "X does not have a shape attribute")

    #if isinstance(C_init,str):
            
    #    if C_init == 'kmeans_plus':
    #        M = KMeans._init_centroids(X,K,init='k-means++', x_squared_norms=x_squared_norms, random_state=random_state)
            #M = _k_init(X, K)
    #        l = km_le(X,M)

    if isinstance(C_init, str):
        if C_init == 'kmeans_plus':
            # Use the public API to create an instance of KMeans with initial centroids
            kmeans = KMeans(n_clusters=K, init='k-means++', random_state=seedValue)
            kmeans.fit(X)
            M = kmeans.cluster_centers_
            l = km_le(X, M)
            
        if C_init =='kmeans':
            pt = timeit.default_timer()
            # print("ID of process running worker: {}".format(os.getpid()))
            kmeans = KMeans(n_clusters=int(K),random_state=int(seedValue)).fit(X)
            pp = timeit.default_timer() -pt
            l =kmeans.labels_
            M = kmeans.cluster_centers_
            #print("Intial Kmeans center are "+str(M))
        elif C_init =='kmedian':
            M = KMeans._init_centroids(X, K, init='random')
            #M = _k_init(X, K)
            l = km_le(X,M)
            
    else:
        M = C_init.copy()
        # l = km_le(X,M)
        l = l_init.copy()
        #print("Centers in km_init"+str(M))
    st = timeit.default_timer()
    del C_init, l_init

    
    return M,l

def restore_nonempty_cluster (X,K,seedValue,oldl,oldC,oldS,ts):
        ts_limit = 2
        C_init = 'kmeans'
        if ts>ts_limit:
            trivial_status = True
            l =oldl.copy()
            C =oldC.copy()
            S = oldS.copy()

        else:

            C,l =  km_init(X,K,seedValue,C_init)
            sqdist = ecdist(X,C,squared=True)
            S = normalize_2(np.exp((-sqdist)))
            trivial_status = False
        
        return l,C,S,trivial_status

def fair_clustering(SHARED_VARS, X, K, u_V, V_list,seedValue, lmbda, fairness = False, method = 'kmeans', C_init = "kmeans_plus",
                    l_init = None, A = None):
    
    """ 
    
    Proposed fairness clustering method
    
    """

    #print("SHARED_VARS DENTRO FAIR CLUSTERING", SHARED_VARS)
    seedValue = int(seedValue)
    np.random.seed(seedValue)
    
    N,D = X.shape
    start_time = timeit.default_timer()
    C,l =  km_init(X, K,seedValue, C_init, l_init)
    
    assert len(np.unique(l)) == K
    ts = 0
    S = []
    E_org = []
    E_cluster = []
    E_fair = []
    E_cluster_discrete = []
    fairness_error = 0.0
    oldE = 1e100

    maxiter = 100
    #print("arriva fino a x_s")
    #print("X prima di X_s", X)
    #X_s = utils.init(X_s = X)
    #X_s = utils.init(X_s=X)
    SHARED_VARS = utils.init(X_s=X)
    #print("X_s", X_s)
    pool = multiprocessing.Pool(1)
    
    if A is not None:
        d = A.sum(axis=1)

    cost_over_iter = []
    for i in range(maxiter):
        oldC = C.copy()
        oldl = l.copy()
        oldS = S.copy()
        
        if i == 0:
            if method == 'kmeans':
                sqdist = ecdist(X,C,squared=True)
                a_p = sqdist.copy()

            if method == 'kmedian':
                sqdist = ecdist(X,C,squared=True)
                a_p = sqdist.copy()

            if method == 'ncut':
                S = get_S_discrete(l,N,K)
                sqdist_list = [KernelBound_k(A, d, S[:,k], N) for k in range(K)]
                sqdist = np.asarray(np.vstack(sqdist_list).T)
                a_p = sqdist.copy()

            
        elif method == 'kmeans':
            tmp_list = [np.where(l==k)[0] for k in range(K)]
            #print("kmeans_update", kmeans_update)
            #print("SHARED_VARS PRIMA DI POOL mAP", X_s.shape)
            #print("tmp_list PRIMA DI POOL mAP", tmp_list, type(tmp_list))
            #kmu = kmeans_update(tmp_list[0], X_s)
            #print("kmu", kmu)
            #args_list = [(X_s, tmp) for tmp in tmp_list]
            #args_list = (tmp_list, SHARED_VARS)
            #print("Arguments list for pool.map:", args_list)
            # Ensure args_list is defined as previously mentioned
            args_list = [(tmp, SHARED_VARS) for tmp in tmp_list]

            # Now, instead of using pool.map, use a simple for loop to process each item in args_list
            C_list = []
            for args in args_list:
                c1 = kmeans_update(*args)  # Unpack each tuple into the kmeans_update function
                C_list.append(c1)
            #print("SHARED_VARS PRIMA DI POOL MAP", X_s)
            #C_list = pool.map(kmeans_update, args_list)
            #C_list = kmeans_update(tmp_list[0], SHARED_VARS)
            #print("DOPO C_LIST ")
            #print("C_list pre C", C_list)
            C = np.asarray(np.vstack(C_list))
            #print("C", C)
            #print("Shapes X C", X.shape, C.shape)
            sqdist = ecdist(X,C,squared=True)
            a_p = sqdist.copy()

        elif method == 'kmedian':
            tmp_list = [np.where(l==k)[0] for k in range(K)]
            C_list = pool.map(kmeans_update(tmp_list[0], SHARED_VARS),tmp_list)
            #print("C_list pre C", C_list)
            C = np.asarray(np.vstack(C_list))
            #print("C", C)
            sqdist = ecdist(X,C)
            a_p = sqdist.copy()

        elif method == 'ncut':
            S = get_S_discrete(l,N,K)
            sqdist_list = [KernelBound_k(A, d, S[:,k], N) for k in range(K)]
            sqdist = np.asarray(np.vstack(sqdist_list).T)
            a_p = sqdist.copy()

        if fairness ==True and lmbda!=0.0:

            l_check = a_p.argmin(axis=1)
            
            # Check for empty cluster
            if (len(np.unique(l_check))!=K):
                l,C,S,trivial_status = restore_nonempty_cluster(X,K,seedValue,oldl,oldC,oldS,ts)
                ts = ts+1
                if trivial_status:
                    break
                
            bound_iterations = 5000

            l,S,bound_E = bound_update(a_p, u_V, V_list, lmbda, bound_iterations)
            fairness_error = get_fair_accuracy_proportional(u_V,V_list,l,N,K)

        else:
                
            if method == 'ncut':
                l = a_p.argmin(axis=1)
                S = get_S_discrete(l,N,K)
            
            else:
                S = get_S_discrete(l,N,K)
                l = km_le(X,C)
            
        currentE, clusterE, fairE, clusterE_discrete = compute_energy_fair_clustering(X, C, l, S, u_V, V_list,lmbda, A = A, method_cl=method)
        E_org.append(currentE)
        E_cluster.append(clusterE)
        E_fair.append(fairE)
        E_cluster_discrete.append(clusterE_discrete)
        
        
        if (len(np.unique(l))!=K) or math.isnan(fairness_error):
            l,C,S,trivial_status = restore_nonempty_cluster(X,K,seedValue,oldl,oldC,oldS,ts)
            ts = ts+1
            if trivial_status:
                break

        cost_over_iter.append(clusterE_discrete)
        if (i>1 and (abs(currentE-oldE)<= 1e-4*abs(oldE))):
            break

        else:       
            oldE = currentE.copy()

    pool.close()
    pool.join()
    pool.terminate()
    elapsed = timeit.default_timer() - start_time
    E = {'fair_cluster_E':E_org,'fair_E':E_fair,'cluster_E':E_cluster, 'cluster_E_discrete':E_cluster_discrete}
    return C,l,elapsed,S,E

#if __name__ == '__main__':
#    main()

    
    