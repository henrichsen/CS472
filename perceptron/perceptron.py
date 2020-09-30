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

    def __init__(self, lr=.1, shuffle=True, deterministic=10):
        """ Initialize class with chosen hyperparameters.

        Args:
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        """
        self.lr = lr
        self.shuffle = shuffle
        self.initial_weights = []
        self.weights = []
        self.deterministic = deterministic

    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        self.initial_weights = self.initialize_weights(X) if initial_weights is None else initial_weights
        self.weights = self.initial_weights
        assert (len(X) == len(y))

        best_weights = self.weights
        value = []
        net = []

        finished = False
        base_accuracy = 0
        counter = 0
        while self.deterministic>0:

            for row, rowy in zip(X, y):

                single_value = np.dot(row, self.weights[:-1]) + self.weights[-1] * 1
                value = np.append(value, [single_value])
                net = np.append(net, 1 if value[-1] > 0 else 0)
                if net[-1] != rowy:
                    under = net[-1] < rowy
                    self.update_weights(under, row)
                current_accuracy = self.score(X, y)
                if current_accuracy > base_accuracy:
                    base_accuracy = current_accuracy
                    best_weights = self.weights
            if self.shuffle:
                X, y = self._shuffle_data(X, y)
            print(str(1-current_accuracy))
            self.deterministic -= 1
        print('Accuracy')
        print(base_accuracy)
        print('best weights')
        print(best_weights)
        return self

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """

        net = []
        for row in X:
            row = np.append(row, 1)
            value = np.dot(row, self.weights) + self.weights[-1] * 1

            net = np.append(net, 1 if value > 0 else 0)
        return net
        pass

    def initialize_weights(self, X=[[]]):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns: initial weights for the given inputs X

        """
        self.initial_weights = np.zeros(len(X[0]) + 1)
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
        net = self.predict(X)
        length = len(y)
        correct = 0
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
        indices = np.arange(length)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        return X, y
        pass

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights
        pass

    def update_weights(self, under, X):

        change = 1 if under else -1
        deltaW = self.lr * change * X
        deltaW = np.append(deltaW, self.lr * change * 1)
        self.weights += deltaW

        pass
