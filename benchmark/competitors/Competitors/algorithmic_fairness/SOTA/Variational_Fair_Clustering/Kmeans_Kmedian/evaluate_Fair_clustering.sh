#!/bin/bash

c_v_l=False # Set true for clusters vs lambda figures in Synthetic or Synthetic-unequal dataset
f_v_E=False # Set true to check the fairness error vs Discrete clustering energy plots in a lambda range. Note that,
          # In this case also set --lmd_tune to True to have the default range.
conv=False # Set true to see if the algorithm converges.
lmd_tune=False
#source fair_clustering/bin/activate
#cd Downloads/Fresh\ Experiments/Kmeans/Variational/

dataset=Diabetes   #diabtest adult done bank done  and bank full on exec 
cluster_option=kmeans
lmd=0 # start this 0.003
K=10 #10,15,20,30,40] start with 30 ) ......<2,5 on exec>, 10,15,20,30 ono exec cenii
#seed=0  #set seed here and in lower cmd 

python test_fair_clustering.py -d $dataset \
                             --cluster_option $cluster_option \
                             --lmbda-tune $lmd_tune \
                             --lmbda $lmd \
                             --K  $K\
                             --plot_option_clusters_vs_lambda $c_v_l \
                             --plot_option_fairness_vs_clusterE $f_v_E \
                             --plot_option_convergence $conv
# 20ka 5th and th give dikat

# dataset=Synthetic-unequal
# cluster_option=kmeans
# lmd=60.0
# python test_fair_clustering.py -d $dataset \
#                              --cluster_option $cluster_option \
#                              --lmbda-tune $lmd_tune \
#                              --lmbda $lmd \
#                              --plot_option_clusters_vs_lambda $c_v_l \
#                              --plot_option_fairness_vs_clusterE $f_v_E \
#                              --plot_option_convergence $conv

#Synthetic
#dataset=Synthetic
#cluster_option=ncut
#lmd=60.0
#python test_fair_clustering.py -d $dataset \
#                             --cluster_option $cluster_option \
#                             --lmbda-tune $lmd_tune \
#                             --lmbda $lmd \
#                             --plot_option_clusters_vs_lambda $c_v_l \
#                             --plot_option_fairness_vs_clusterE $f_v_E \
#                             --plot_option_convergence $conv
##Census II
#dataset=CensusII
#cluster_option=kmeans
#lmd=10000
#python test_fair_clustering.py -d $dataset \
#                             --cluster_option $cluster_option \
#                             --lmbda-tune $lmd_tune \
#                             --lmbda $lmd \
#                             --plot_option_clusters_vs_lambda $c_v_l \
#                             --plot_option_fairness_vs_clusterE $f_v_E \
#                             --plot_option_convergence $conv


