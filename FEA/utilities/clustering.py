# """
# Clustering algorithms.
# These algorithms will be used for clustering features/attributes.
# Three different algorithms:
#     1. Fuzzy K-Means (baseline)
#     3. HDBSSCAN, maybe one day
# """
#
# import numpy as np
# import time, skfuzzy
# from abc import ABC, abstractmethod
# import copy
# # from minepy import MINE
# from deap.benchmarks import *
# from networkx.convert_matrix import *
# import networkx as nx
# import matplotlib.pyplot as plt
# import scipy as sp
# # import hdbscan
#
# class Cluster(ABC):
#     '''
#     Abstract class for the various clustering techniques
#     test
#     '''
#
#     def __init__(self, data):
#         self.data = data
#         super().__init__()
#
#     @abstractmethod
#     def assign_clusters(self):
#         '''
#         Abstract method to assign data points to clusters
#         implementation varies by method. Returns the cluster
#         assignments as an array.
#         '''
#
#         pass
#
#     def return_factors(self, cluster_probs, threshold):
#         factors = []
#         outliers = np.arange(len(cluster_probs[0, :]))
#         for i in range(len(cluster_probs)): # number of clusters?
#             cluster = []
#             for j in range(len(cluster_probs[i, :])):
#                 if cluster_probs[i, j] > threshold:
#                     cluster.append(j)
#
#             factors.append(tuple(cluster))
#         outliers = set(outliers) - set.union(*map(set, factors))
#         if outliers:
#             factors.append(tuple(outliers))
#
#         return factors, outliers
#
#
# class FuzzyKmeans(Cluster):
#     '''
#     Implementation of the fuzzy k-means clustering algorithm using
#     the scikit learn fuzzy implementation
#     '''
#
#     def __init__(self, data, k=3):
#         self.k = k
#         self.soft_clusters = None
#         super().__init__(data)
#
#     def assign_clusters(self):
#         cntr, u, u0, d, jm, p, fpc = skfuzzy.cluster.cmeans(self.data.T, self.k, 2, error=0.005, maxiter=1000,
#                                                             init=None)
#         # I think u_pred = u from line above since using same data, so next line not needed
#         #u_pred, u0, d, jm, p, fpc = skfuzzy.cluster.cmeans_predict(self.data.T, cntr, 2, error=0.005, maxiter=1000)
#         self.soft_clusters = u
#         return self.soft_clusters