import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import math


class KNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, columntype=[], weight_type='inverse_distance', k=-1, normalize=False):  ## add parameters here
        """
        Args:
            columntype for each column tells you if continues[real]==0 or if nominal[categoritcal]==-1.
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
        """
        assert columntype.ndim == 1
        self.columntype = columntype  # Note This won't be needed until part 5

        assert weight_type == "no_weight" or weight_type == "inverse_distance"
        self.weight_type = weight_type
        assert type(k) == int
        self.k = k
        assert type(normalize) == bool
        self.normal = normalize
        self.Inputs = None
        self.Outputs = None

    def fit(self, X, y):
        """ Fit the data; run the algorithm (for this lab really just saves the data :D)
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """

        self.Inputs = X
        self.Outputs = y
        return self

    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        if self.normal:
            X = self.normalize(X)
            print('Normal')
        else:
            self.Inputs_n =self.Inputs
        if self.k <= 0 or self.k > len(X):
            self.k = len(X)
        y = np.zeros([len(X), len(np.unique(self.Outputs))])
        ymax = np.zeros(len(y))
        for row in range(len(X)):
            distances = self.calculate_distance(X[row], self.Inputs_n)
            sort = np.argpartition(distances,self.k)
            distance = distances[sort]
            outputs = self.Outputs[sort]
            for k in range(self.k - 1):
                value = int(outputs[k])
                y[row][value] += self.calculate_weight(distance[k])
            ymax[row] = np.argmax(y[row])
        print('predicted')
        return ymax

        pass

        # Returns the Mean score given input data and labels
    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
                X (array-like): A 2D numpy array with data, excluding targets
                y (array-like): A 2D numpy array with targets
        Returns:
                score : float
                        Mean accuracy of self.predict(X) wrt. y.
        """
        guess = self.predict(X)
        # np.savetxt("evaluation_output.csv", guess, delimiter=",") # save guess
        count = 0
        for row in range(len(y)):
            if (y[row] == guess[row]):
                count += 1
        return count / len(y)

    def calculate_distance(self, input, array):
        assert input.ndim == 1
        assert array.ndim == 2
        assert len(input) == len(array[0])
        distance = np.zeros([len(array)])

        for row in range(len(array)):
            squared_distance = 0
            for column in range(len(input)):
                if np.isnan(array[row][column]):  # if value unknown
                    distance[row] += 1
                elif self.columntype[column] == 0:  # if value is real
                    squared_distance += (input[column] - array[row][column]) ** 2
                elif self.columntype[column] == -1:  # if value is categorical
                    if input[column] == array[row][column]:
                        distance[row] += 0
                    else:
                        distance[row] += 1
            distance[row] += math.sqrt(squared_distance)

        return distance

    def calculate_weight(self, distance):
        if self.weight_type == 'no_weight':
            return 1
        elif self.weight_type == 'inverse_distance':
            return 1 / distance

    def normalize(self, array2d):
        max = np.zeros(len(array2d[0]))
        min = np.zeros(len(array2d[0]))

        for column in range(len(array2d[0])):
            if self.columntype[column] == -1:
                continue
            max[column] = np.max([np.max(array2d[:, column]), np.max(self.Inputs[:,column])])
            min[column] = np.max([np.min(array2d[:, column]), np.min(self.Inputs[:,column])])
            for row in range(len(array2d)):
                array2d[row][column] = (array2d[row][column] - min[column]) / (max[column] - min[column])
                self.Inputs_n[row][column] = (self.Inputs[row][column] - min[column]) / (max[column] - min[column])

        return array2d
