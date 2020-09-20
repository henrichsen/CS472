import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

from sklearn.linear_model import Perceptron


class PerceptronClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, lr=.1, shuffle=True):
        """ Initialize class with chosen hyperparameters.

        Args:
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        """
        self.lr = lr
        self.shuffle = shuffle

    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        self.initial_weights = self.initialize_weights() if not initial_weights else initial_weights

        assert (len(X) == len(y))
        assert (len(X[0]) + 1 == len(initial_weights))

        weights = initial_weights
        best_weights = weights
        value = []
        net = []

        finished = False
        base_accuracy = 0
        counter = 0
        while not finished:
            for row, rowy in zip(X, y):
                row.append(1)

                for column in row:
                    value.append(np.dot(column,weights))

                net.append(1 if value[-1] > 0 else 0)
                if net[-1] != rowy:
                    under = net[-1] < rowy
                    weights = self.update_weights(self, under, weights, X)
            current_accuracy = self.score(self, X, y)
            if current_accuracy > base_accuracy:
                base_accuracy = current_accuracy
                best_weights = weights
                counter = 0
            else:
                counter += 1
            if counter >= 5:
                finished = True
                weights = best_weights
            if self.shuffle:
                self._shuffle_data(self, X, y)
        return self

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        value = 0
        net = 0
        for row in X:
            row.append(1)

            for column in row:
                value.append(np.dot(column, self.initial_weights))

            net.append(1 if value[-1] > 0 else 0)
        return net
        pass

    def initialize_weights(self, X=[[]]):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns: initial weights for the given inputs X

        """
        self.initial_weights = np.zeros(len(X[0])+1)
        return self.initial_weights

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        net = self.predict(self, X)
        length = len(y)
        correct = 0;
        for valuen, valuey in zip(net, y):
            if valuen == valuey:
                correct += 1
        return correct / length

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        length = len(y)
        index = 0
        indices = np.arange(length)
        np.random.shuffle(indices,dtype=int)
        X = X[indices]
        y = y[indices]

        pass

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        pass

    def update_weights(self, under, weights, X):

        change = 1 if under else -1
        for w, x in zip(weights, X):
            w += self.lr*change*x

        return weights

        pass
