import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified.


class MLPClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, lr=.1, momentum=0, shuffle=True, hidden_layer_widths=None):
        """ Initialize class with chosen hyperparameters.

        Args:
            lr (float): A learning rate / step size.
            shuffle(boolean): Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
            momentum(float): The momentum coefficent 
        Optional Args (Args we think will make your life easier):
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer if hidden layer is none do twice as many hidden nodes as input nodes.
        Example:
            mlp = MLPClassifier(lr=.2,momentum=.5,shuffle=False,hidden_layer_widths = [3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        self.hidden_layer_widths
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle

    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Optional Args (Args we think will make your life easier):
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        #self.initial_weights = self.initialize_weights() if initial_weights is None else initial_weights

        length = 2 * len(X) if self.hidden_layer_widths is None else self.hidden_layer_width

        ones = np.ones((len(X), 1))
        X = np.concatenate((X, ones), axis=1) # add bias
        width = len(X)
        hidden_layer_weights = np.random.rand(-1, 1, (length, width))

        length = len(y)
        width = len(hidden_layer_weights)+1 # include bias
        output_layer_weights = np.random.uniform(-1, 1, (length, width))
        
        if initial_weights is not None:
            hidden_layer_weights = self.initialize_weights(hidden_layer_weights, initial_weights['hidden'])
            output_layer_weights = self.initialize_weights(output_layer_weights, initial_weights['output'])
        hidden_layer_net= len(hidden_layer_weights)
        output_layer_net= len(output_layer_weights)
        hidden_layer_z= len(hidden_layer_weights)
        output_layer_z= len(output_layer_weights)

        
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

    def initialize_weights(self, weights=None, initial_weights=None):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        assert len(weights) == len(initial_weights)
        assert len(weights[0]) == len(initial_weights[0])
        return initial_weights

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """

        return 0

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        indices = np.arrange(len(y))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        return X, y
        pass

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        pass

    def fnet(self, net):
        Zj = 1 / (1 + exp(-net))
        return Zj
        pass

    def fprime(self, net):
        Z = self.fnet(Zj)
        return Z * (1 - Z)
        pass

    def OutDelta(self, net, target):
        return (target - self.fnet(net)) * self.fprime(net)
        pass

    def HiddenDelta(self, net, odelta, w):
        sumj = 0
        for deltak, weightk in zip(odelta, w):
            sumj += deltak * weightk
        sumj *= self.fprime(net)
        return sumj
        pass
