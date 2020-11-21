import numpy as np
import random
from sklearn.base import BaseEstimator, ClusterMixin


class KMEANSClustering(BaseEstimator, ClusterMixin):

    def __init__(self, k=3, debug=False):  ## add parameters here
        """
        Args:
            k = how many final clusters to have
            debug = if debug is true use the first k instances as the initial centroids otherwise choose random points as the initial centroids.
        """
        self.k = k
        self.debug = debug

    def fit(self, X, y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        assert len(X) >= self.k

        centroids = np.zeros([self.k, len(X[0,:])])
        if self.debug:
            centroids = X[:self.k, :]
        else:
            for i in range(len(centroids)):
                newcentroid = X[random.randint(0, len(X))]
                if newcentroid in centroids:
                    i -= 1
                else:
                    centroids[i] = newcentroid
        print(centroids)
        clusterid = np.zeros(len(X))
        for row in range(len(X)):
            distance_2_centroids = np.zeros(self.k)
            for centroid_index in range(len(centroids)):
                distance_2_centroids[centroid_index] = self.calculate_distance(X[row], centroids[centroid_index])
            clusterid[row] = np.argmin(distance_2_centroids)
        print(clusterid)
        return self

    def save_clusters(self, filename):
        """
            f = open(filename,"w+") 
            Used for grading.
            write("{:d}\n".format(k))
            write("{:.4f}\n\n".format(total SSE))
            for each cluster and centroid:
                write(np.array2string(centroid,precision=4,separator=","))
                write("\n")
                write("{:d}\n".format(size of cluster))
                write("{:.4f}\n\n".format(SSE of cluster))
            f.close()
        """

    def calculate_distance(self, X, Y):
        assert len(X) == len(Y)
        distance = 0
        for i in range(len(X)):
            distance += (X[i] - Y[i]) ** 2
        return distance ** .5

    def calculate_centroid(self, X):
        centroid = np.zeros([1, len(X[:])])
        for column in range(X[:]):
            for row in range(X):
                centroid[1, column] += X[row, column]
            centroid[1, column] /= len(X)
        return centroid
