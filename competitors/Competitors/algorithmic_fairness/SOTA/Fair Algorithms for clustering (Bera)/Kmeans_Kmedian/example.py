#! /home/hpc128/fair_clustering/bin/python
import configparser
import sys
import numpy as np
import random 
from fair_clustering import fair_clustering
from util.configutil import read_list

config_file = "config/example_config.ini"
config = configparser.ConfigParser(converters={'list': read_list})
config.read(config_file)

# Create your own entry in `example_config.ini` and change this str to run
# your own trial
config_str = "self_bank3"
print("Using config_str = {}".format(config_str))

# Read variables
data_dir = config[config_str].get("data_dir")
dataset = config[config_str].get("dataset")
clustering_config_file = config[config_str].get("config_file")
num_clusters = list(map(int, config[config_str].getlist("num_clusters")))
deltas = list(map(float, config[config_str].getlist("deltas")))
max_points = config[config_str].getint("max_points")
violating = config["DEFAULT"].getboolean("violating")
violation = config["DEFAULT"].getfloat("violation")

seeds = [0,100,200,300,400,500,600,700,800,900,1000,1100]
for n_clusters in num_clusters:
    print("K =="+str(n_clusters))
    for run in range(0,10):
        np.random.seed(seeds[run])
        random.seed(seeds[run])
        print("*** Run "+str(run)+" ******") 
        seedValue = seeds[run]
        fair_clustering(dataset, clustering_config_file, seedValue,data_dir+dataset+"_K_"+str(n_clusters)+"_", n_clusters, deltas, max_points, violating, violation)
