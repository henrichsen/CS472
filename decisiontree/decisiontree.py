import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score

class DTClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, counts=None):
        """ Initialize class with chosen hyperparameters.
        Args:
        Optional Args (Args we think will make your life easier):
            counts: A list of Ints that tell you how many types of each feature there are
        Example:
            DT  = DTClassifier()
            or
            DT = DTClassifier(count = [2,3,2,2])
            Dataset = 
            [[0,1,0,0],
            [1,2,1,1],
            [0,1,1,0],
            [1,2,0,1],
            [0,0,1,1]]

        """
        self.tree = dict()

    def fit(self, X, y):
        """ Fit the data; Make the Desicion tree

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        if len(X[0]) == 0:  # if no more attributes return most common output
            value, count = np.unique(y, return_counts=True)
            return value[np.argmax(count)]
        if self.info(y) == 0:  # if node is pure return output
            return y[0]

        # if neither condition split by attribute

        split_index = self.get_best_attribute(X, y)
        self.tree[split_index] = self.get_values(X, split_index)
        for value in (self.tree[split_index]):
            self.tree[split_index][value] = self.fit(*self.split_by_value(X, y, split_index, value))
        print(self.tree)
        return self

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        pass

    def score(self, X, y):
        """ Return accuracy(Classification Acc) of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array of the targets 
        """
        return np.sum(self.predict(X) == y) / len(y)

    def get_best_attribute(self, X, y):
        # return index of best attribute to split
        info_array = np.zeros(len(X[0]))
        for i in range(len(X[0, :])):
            info_array[i] = self.info_A(X[:, i], y)

        least_info = int(np.argmin(info_array))
        return least_info

    def info_A(self, attribute_list, y):
        # return the information by splitting by an attribute
        info = 0
        values = np.unique(attribute_list)

        for A in values:
            yj = np.array(0)
            boolean_A = attribute_list == A
            for b_a, y_ in zip(boolean_A, y):
                if b_a:
                    yj = np.append(yj, y_)
            info += len(yj) / len(attribute_list) * self.info(yj)

        return info

    def info(self, yj):
        outputs = np.unique(yj)
        sum = 0

        for C in outputs:
            percent = np.sum(yj == C) / len(yj)
            if percent != 0:
                sum += np.log2(percent) * percent
        return -sum

    def split_by_value(self, X, y, index, value):
        divided_X = np.array(1)
        divided_Y = np.array(1)
        same_value = (value == X[:, index])
        for i in range(len(same_value)):
            if same_value[i]:
                divided_X = np.append(divided_X, X[i])
                divided_Y = np.append(divided_Y, y[i])
        return divided_X, divided_Y

    def get_values(self, X, index):
        return np.unique(X[:, index])


"""
def resursive(X,y)

while len(X[0]) and info(y) != 0:
    split =self.split_by_attribute(X,y,self.get_best_attribute(X,y))
    for s in split
        self.recursive(s[0],s[1)
"""
