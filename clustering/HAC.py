import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class HACClustering(BaseEstimator, ClusterMixin):

    def __init__(self, k=3, link_type='single', print_k=[]):  ## add parameters here
        """
        Args:
            k = how many final clusters to have
            link_type = single or complete. when combining two clusters use complete link or single link
        """
        assert link_type == 'single' or link_type == 'complete'
        self.link_type = link_type
        self.k = k
        if not print_k:
            self.print_k = [self.k]
        else:
            self.print_k = print_k

    def fit(self, X, y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        assert self.k < len(X)

        ## start calculate SSE
        data_centroid = self.calculate_centroid(X)[0]
        distance_calc = np.zeros(len(X))
        for j in range(len(X)):
            distance_calc[j] = self.calculate_distance(data_centroid, X[j]) ** 2
        print('Total SSE: ' + str(np.sum(distance_calc)))
        ## end calculate SSE
        ## start calculate distances
        distances = np.zeros([len(X), len(X)])
        for row in range(len(X)):
            for column in range(len(X)):
                distances[row][column] = self.calculate_distance(X[row], X[column])
        np.fill_diagonal(distances, np.inf)
        ## end calculate distances
        cluster_id = np.arange(0, len(X), 1, dtype=int)

        while len(np.unique(cluster_id)) > self.k:
            min_row, min_column = np.unravel_index(np.argmin(distances), distances.shape)
            old_id = cluster_id[min_column]
            new_id = cluster_id[min_row]
            cluster_id[cluster_id == old_id] = new_id
            if self.link_type == 'single':
                distances[new_id] = np.min(distances[new_id, old_id], axis=0)
                distances[new_id, new_id] = np.inf
            elif self.link_type == 'complete':
                distances[new_id] = np.max(distances[new_id, old_id], axis=0)
            else:
                assert False
            distances[:, new_id] = distances[new_id]
            distances[old_id] = np.inf
            distances[:, old_id] = np.inf
            ## Start print
            if np.any(np.isin(self.print_k, len(np.unique(cluster_id)))):
                print('K=' + str(len(np.unique(cluster_id))))
                for unique in np.unique(cluster_id):
                    values = X[unique == cluster_id]
                    print("Centroid: ")
                    centroid_value = self.calculate_centroid(values)[0]
                    print(centroid_value)
                    print('Number of instances in Cluster: ' + str(len(values)))
                    distance_calc = np.zeros(len(values))
                    for j in range(len(values)):
                        distance_calc += self.calculate_distance(centroid_value, values[j]) ** 2
                    print("SSE: " + str(np.sum(distance_calc)))

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
