# k_means.py
# Theta Technologies
########################################################################################################################
# Driver function for facilitating from-scratch K-means clustering over multimodal sensor data.
########################################################################################################################

from __future__ import division, print_function
from sklearn import datasets

from Alchemy.supervised_learning.k_means import KMeans
from Alchemy.utils import Plot


def build_k_means_clustering(datapath:str, plot_results:bool=True):
    # Dummy data
    X, y = datasets.make_blobs()

    # Cluster the data using K-Means
    clf = KMeans(k=3)
    y_pred = clf.predict(X)

    # Project the data onto the 2 primary principal components
    p = Plot()
    p.plot_in_2d(X, y_pred, title="K-Means Clustering")
    p.plot_in_2d(X, y, title="Actual Clustering")



if __name__ == "__main__":
    build_k_means_clustering("./")