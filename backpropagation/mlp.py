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
        # start initailize data
        length = 2 * len(X) if self.hidden_layer_widths is None else self.hidden_layer_width

        ones = np.ones((len(X), 1))
        X = np.concatenate((X, ones), axis=1) # add bias
        width = len(X)
        hidden_layer_weights = np.random.rand(-1, 1, (length, width))
        hidden_layer_old_delta_weights = np.zeros(length, width)


        length = len(y)
        width = len(hidden_layer_weights)+1 # include bias
        output_layer_weights = np.random.uniform(-1, 1, (length, width))
        output_layer_old_delta_weights= np.zeros(length, width)
        if initial_weights is not None:
            hidden_layer_weights = self.initialize_weights(hidden_layer_weights, initial_weights['hidden'])
            output_layer_weights = self.initialize_weights(output_layer_weights, initial_weights['output'])

        # end initailize data

        for x in X:
            hidden_layer_net, output_layer_net, hidden_layer_z, output_layer_z, output_layer_z_binary = self.calculate_Z(hidden_layer_weights, x, output_layer_weights)


        # start Calculate delta weights
        hidden_layer_delta_weights, output_layer_delta_weights =self.calculate_delta_weights(hidden_layer_z,output_layer_z,hidden_layer_net,output_layer_net,hidden_layer_weights,output_layer_weights,x, y)

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
        Zj = 1 / (1 + np.exp(-net))
        return Zj
        pass

    def fprime(self, net):
        Z = self.fnet(net)
        return Z * (1 - Z)
        pass

    def output_delta(self,T,z,net):
        return (T-z)*self.fprime(net)
        pass

    def calculate_Z(self,hidden_layer_weights, x, output_layer_weights):
        hidden_layer_net = np.zeros(len(hidden_layer_weights))
        output_layer_net = np.zeros(len(output_layer_weights))
        hidden_layer_z = np.zeros(len(hidden_layer_weights))
        output_layer_z = np.zeros(len(output_layer_weights))
        output_layer_z_binary = np.zeros(len(output_layer_weights))

        for z, net, weights in zip(hidden_layer_z, hidden_layer_net, hidden_layer_weights):
            net = np.dot(x, weights)
            z = self.fnet(net)
        for z, net, weights,  input in zip(output_layer_z, output_layer_net, output_layer_weights, hidden_layer_z):
            net = np.dot(input, weights)
            z = self.fnet(net)
        output_layer_z_binary = np.around(output_layer_z)

        return hidden_layer_net, output_layer_net, hidden_layer_z, output_layer_z, output_layer_z_binary

    def calculate_delta(self,hidden_layer_z,output_layer_z,hidden_layer_net,output_layer_net,hidden_layer_weights, y):

        output_layer_delta = np.zeros(len(output_layer_z))
        hidden_layer_delta = np.zeros(len(hidden_layer_z))
        # calculate output layer delta
        for target, z, net, delta in zip(y, output_layer_z, output_layer_net, output_layer_delta):
            delta = self.output_delta(target, z, net)
        # calculate hidden layer delta
        for delta, weights, net in zip(output_layer_delta, hidden_layer_weights, hidden_layer_net):
            sum = 0
            for w in weights:
                sum += delta * w
            delta = sum * self.fprime(net)

        return hidden_layer_delta, output_layer_delta

    def calculate_delta_weights(self,hidden_layer_z,output_layer_z,hidden_layer_net,output_layer_net,hidden_layer_weights,output_layer_weights,x, y,hidden_layer_old_delta_weights,output_layer_old_delta_weights):

        hidden_layer_delta_weights = hidden_layer_weights
        output_layer_delta_weights = output_layer_weights

        output_layer_delta, hidden_layer_delta =self.calculate_delta(hidden_layer_z,output_layer_z,hidden_layer_net,output_layer_net,hidden_layer_weights, y,)

        for delta_weights, delta, old_delta_weights in zip(output_layer_delta_weights,  output_layer_delta, output_layer_old_delta_weights):
            for delta_weight, z, old_delta_weight in zip(delta_weights, hidden_layer_z, old_delta_weights):
                delta_weight = self.lr*delta*z+self.momentum*old_delta_weight
        for delta_weights, delta, old_delta_weights in zip(hidden_layer_delta_weights, hidden_layer_delta, hidden_layer_old_delta_weights):
            for delta weight, z, old_delta_weight in zip(delta_weights, x, old_delta_weights): ## only current row of X
                delta_weight = self.lr*delta*z+self.momentum*old_delta_weight
                
        return hidden_layer_delta_weights, output_layer_delta_weights