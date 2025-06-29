# -*- coding: utf-8 -*-

import timeit
import numexpr as ne

from benchmark.competitors.Competitors.algorithmic_fairness.SOTA.Variational_Fair_Clustering.Kmeans_Kmedian.src.progressBar import \
    printProgressBar


#import random
#random.seed(0)
#np.random.seed(0)



#@jit(parallel=True)
def compute_b_j_parallel(J,S,V_list,u_V):
    result = [compute_b_j(V_list[j],u_V[j],S) for j in range(J)]
    return result


def compute_b_j(V_j,u_j,S_):
    N,K = S_.shape
    V_j = V_j.astype('float')
    S_sum = S_.sum(0)
    R_j = ne.evaluate('u_j*(1/S_sum)')
    F_j_a = np.tile((ne.evaluate('u_j*V_j')),[K,1]).T
    F_j_b = np.maximum(np.tile(np.dot(V_j,S_),[N,1]),1e-15)
    F_j = ne.evaluate('R_j - (F_j_a/F_j_b)')

    return F_j


#@jit
def get_S_discrete(l,N,K):
    x = range(N)
    temp =  np.zeros((N,K),dtype=float)
    temp[(x,l)]=1
    return temp

def normalize_2(S_in):
    S_in_sum = S_in.sum(1)[:,np.newaxis]
    # S_in = np.divide(S_in,S_in_sum)
    S_in = ne.evaluate('S_in/S_in_sum')
    return S_in


import numexpr as ne


def normalize_2(S_in):
    S_in_sum = S_in.sum(1)[:, np.newaxis]  # Sum along rows
    S_in_sum[S_in_sum == 0] = 1e-10  # Prevent division by zero
    S_in = ne.evaluate('S_in / S_in_sum')  # Perform element-wise division
    #for k in S_in:
    #    print(k)
    return S_in



def normalize(S_in):
    maxcol = S_in.max(1)[:, np.newaxis]  # Get max value for each row
    S_in = ne.evaluate('S_in - maxcol')  # Subtract max value for numerical stability

    S_out = np.exp(S_in)  # Compute exponential
    S_out_sum = S_out.sum(1)[:, np.newaxis]  # Sum along rows

    # Prevent division by zero
    S_out_sum[S_out_sum == 0] = 1e-10  # Ensures no zero denominator

    S_out = ne.evaluate('S_out / S_out_sum')  # Normalize each row
    return S_out


def bound_energy(S, S_in, a_term, b_term, L, bound_lambda, batch = False):

    E = np.nansum((S*np.log(np.maximum(S,1e-15)) - S*np.log(np.maximum(S_in,1e-15)) + a_term*S + b_term*S))


    return E
import numpy as np



def bound_energy(S, S_in, a_term, b_term, L, bound_lambda, batch=False):
    """Computes the bound energy for the optimization step with debug checks."""

    # Step 1: Check for NaN or Inf in inputs
    #print("Checking input matrices for NaN or Inf values...")
    #print("NaN in S:", np.isnan(S).any(), " | Inf in S:", np.isinf(S).any())
    #print("NaN in S_in:", np.isnan(S_in).any(), " | Inf in S_in:", np.isinf(S_in).any())
    #print("NaN in a_term:", np.isnan(a_term).any(), " | Inf in a_term:", np.isinf(a_term).any())
   # print("NaN in b_term:", np.isnan(b_term).any(), " | Inf in b_term:", np.isinf(b_term).any())

    # Step 2: Ensure non-negative values
    #print("Min value of S:", np.min(S), " | Max value of S:", np.max(S))
    #print("Min value of S_in:", np.min(S_in), " | Max value of S_in:", np.max(S_in))

    # Step 3: Clip values to prevent log(0)
    S_safe = np.clip(S, 1e-15, None)
    S_in_safe = np.clip(S_in, 1e-15, None)

    # Step 4: Compute logs separately
    log_S = np.log(S_safe)
    log_S_in = np.log(S_in_safe)
    
    #print("NaN in log(S_safe):", np.isnan(log_S).any())
   # print("NaN in log(S_in_safe):", np.isnan(log_S_in).any())

    # Step 5: Compute terms separately
    term1 = S_safe * log_S
    term2 = S_safe * log_S_in
    term3 = a_term * S
    term4 = b_term * S

   # print("NaN in term1 (S * log(S)):", np.isnan(term1).any())
   # print("NaN in term2 (S * log(S_in)):", np.isnan(term2).any())
   # print("NaN in term3 (a_term * S):", np.isnan(term3).any())
   # print("NaN in term4 (b_term * S):", np.isnan(term4).any())

    # Step 6: Compute final bound energy
    E = np.nansum(term1 - term2 + term3 + term4)
    
    #print("Final computed energy:", E)

    return E


            
def bound_update(a_p, u_V, V_list, bound_lambda, bound_iteration = 200, debug=False):
    
    """
    """
    start_time = timeit.default_timer()
    print("Inside Bound Update . . .")
    N,K = a_p.shape
    oldE = float('inf')
    J = len(u_V)
    

# Initialize the S
    S = np.exp((-a_p))
    S = normalize_2(S) # normalize_2 gives me nan values why
    L = 2.0

    for i in range(bound_iteration):
        printProgressBar(i + 1, bound_iteration,length=12)
        # S = np.maximum(S, 1e-20)
        S_in = S.copy()
        
        # Get a and b 
        terms = - a_p.copy()

        b_j_list = compute_b_j_parallel(J,S,V_list,u_V)
        b_j_list = sum(b_j_list)
        print('bjlist', b_j_list)
        b_term = ne.evaluate('bound_lambda * b_j_list')
        terms = ne.evaluate('(terms - b_term)/L')
        S_in_2 = normalize(terms)  
        S = ne.evaluate('S_in * S_in_2')
        S = normalize_2(S)
      #  print(S)
        if debug:
            print('b_term = {}'.format(b_term[0:10]))
            print('a_p = {}'.format(a_p[0:10]))
            print('terms = {}'.format(terms[0:10]))
            print('S = {}'.format(S[0:10]))
            #Check for trivial solutions
            l = np.argmax(S,axis=1)
            if len(np.unique(l))<S.shape[1]:
                S = S_in.copy()

       # print('S', S)
      #  print('__')
       # print('SIN', S_in)
        E = bound_energy(S, S_in, a_p, b_term, L, bound_lambda)
        # print('Bound Energy {} at iteration {} '.format(E,i))
        report_E = E
        
        if (i>1 and (abs(E-oldE)<= 1e-5*abs(oldE))):
            print('Converged')
            break

        else:
            oldE = E; report_E = E

    elapsed = timeit.default_timer() - start_time
    print('\n Elapsed Time in bound_update', elapsed)
    l = np.argmax(S,axis=1)
    
    return l,S,report_E
