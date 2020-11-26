import numpy as np
import random
from sklearn.base import BaseEstimator, ClusterMixin


class KMEANSClustering(BaseEstimator, ClusterMixin):

    def __init__(self, k=3, debug=False, tolerance=.000001):  ## add parameters here
        """
        Args:
            k = how many final clusters to have
            debug = if debug is true use the first k instances as the initial centroids otherwise choose random points as the initial centroids.
        """
        self.k = k
        self.debug = debug
        self.centroids = None
        self.tolerance = tolerance

    def fit(self, X, y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        assert len(X) > self.k

        centroids = np.zeros([self.k, len(X[0, :])])
        if self.debug:
            centroids = X[:self.k, :]
        else:
            initial_centroid_index = np.zeros(len(centroids))
            for i in range(len(centroids)):
                rand = random.randint(0, len(X) - 1)
                newcentroid = X[rand]
                if np.any(initial_centroid_index == rand):
                    i -= 1
                else:
                    centroids[i] = newcentroid
        repeat = True
        i = 0
        while repeat:
            #print(i)
            i += 1
            repeat = False
            clusterid = np.zeros(len(X))
            for row in range(len(X)):
                distance_2_centroids = np.zeros(self.k)
                for centroid_index in range(len(centroids)):
                    distance_2_centroids[centroid_index] = self.calculate_distance(X[row], centroids[centroid_index])
                clusterid[row] = np.argmin(distance_2_centroids)
            for centroid_number in range(len(centroids)):
                selection = clusterid == centroid_number
                new_centroid = self.calculate_centroid(X[selection])
                if self.compare_centroid(new_centroid, centroids[centroid_number]):
                    repeat = True
                centroids[centroid_number] = new_centroid
        self.centroids = centroids
        print('Total SSE: ')
        data_centroid = np.zeros(len(X[0]))
        data_centroid = self.calculate_centroid(X)[0]
        distances = np.zeros(len(X))
        for j in range(len(X)):
            distances[j] = self.calculate_distance(data_centroid, X[j, :])
        print(np.sum(distances ** 2))
        print('Centroids: ')
        print(centroids)
        print('Instances in each centroid: ')
        for i in range(len(np.unique(clusterid))):
            print(np.count_nonzero(clusterid == clusterid[i]))
        print('SSE in each Cluster: ')
        for i in range(len(centroids)):
            values = X[clusterid == i]
            distances = np.zeros(len(values))
            for j in range(len(values)):
                distances[j] = self.calculate_distance(centroids[i], values[j])
            print(np.sum(distances ** 2))
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
        assert len(X) > 0
        centroid = np.zeros([1, len(X[0])])
        for column in range(len(X[0])):
            for row in range(len(X)):
                centroid[0, column] += X[row, column]
            centroid[0, column] /= len(X)
        return centroid

    def compare_centroid(self, X, Y):
        y = np.zeros([1, len(Y)])
        y[0] = Y
        Y = y
        sum = 0
        for i in range(len(X[0])):
            sum += (X[0, i] - Y[0, i]) ** 2
        if sum ** .5 < self.tolerance:
            return False
        else:
            return True
