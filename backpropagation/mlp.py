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
        self.hidden_layer_widths = hidden_layer_widths
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
        # self.initial_weights = self.initialize_weights() if initial_weights is None else initial_weights
        # start initailize data
        if self.shuffle:
            X, y = self._shuffle_data(X, y)
        divide = int(.75 * len(X))
        ValidationX = X[divide:]
        Validationy = y[divide:]
        X = X[:divide]
        y = y[:divide]

        length = 2 * len(X[0]) if self.hidden_layer_widths is None else self.hidden_layer_widths

        ones = np.ones((len(X), 1))
        X = np.concatenate((X, ones), axis=1)  # add bias
        width = len(X[0])
        hidden_layer_weights = np.random.rand(length, width) * 2 - 1
        hidden_layer_old_delta_weights = np.zeros((len(hidden_layer_weights), len(hidden_layer_weights[0])))

        length = len(y[0])
        width = len(hidden_layer_weights) + 1  # include bias
        output_layer_weights = np.random.uniform(-1, 1, (length, width)) * 2 - 1
        output_layer_old_delta_weights = np.zeros((length, width))
        if initial_weights is not None:
            hidden_layer_weights = self.initialize_weights(hidden_layer_weights, initial_weights['hidden'])
            output_layer_weights = self.initialize_weights(output_layer_weights, initial_weights['output'])
        epoch_number = 0
        best_accuracy = 0
        iterations_without_improvement = 0

        # end initailize data
        while iterations_without_improvement < 25:
            ## inside while statement
            epoch_number += 1
            for x, y_ in zip(X, y):
                hidden_layer_net, output_layer_net, hidden_layer_z, output_layer_z, output_layer_z_binary = self.calculate_Z(
                    hidden_layer_weights, x, output_layer_weights)
                # start calculate new weights
                hidden_layer_delta_weights, output_layer_delta_weights = self.calculate_delta_weights(hidden_layer_z,
                                                                                                      output_layer_z,
                                                                                                      hidden_layer_net,
                                                                                                      output_layer_net,
                                                                                                      hidden_layer_weights,
                                                                                                      output_layer_weights,
                                                                                                      x, y_,
                                                                                                      hidden_layer_old_delta_weights,
                                                                                                      output_layer_old_delta_weights)
                hidden_layer_weights += hidden_layer_delta_weights
                output_layer_weights += output_layer_delta_weights
                hidden_layer_old_delta_weights = np.array(hidden_layer_delta_weights, copy=True)
                output_layer_old_delta_weights = np.array(output_layer_delta_weights, copy=True)
                #print(hidden_layer_weights)
                #tempX = X[:, :-1]
                #current_accuracy = self.score(tempX, y, hidden_layer_weights, output_layer_weights)
                #print(current_accuracy)
            # end calculate new weights
            ##debug to here
            # start stopping criteria
            current_accuracy = self.score(ValidationX, Validationy, hidden_layer_weights, output_layer_weights)
            test_accuracy = self.score(X[:,:-1], y, hidden_layer_weights, output_layer_weights)
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
            # end stopping criteria
            if self.shuffle:
                X, y = self._shuffle_data(X, y)

            #print('Epoch number: ' + str(epoch_number))
            print('Epoch number: ' + str(epoch_number) + ' With a validation accuracy of: ' + str(current_accuracy)+' and a training accuracy of: '+str(test_accuracy))  ## for testing
            print('Hidden layer weights: ')
            print(hidden_layer_weights)
            print('Output layer weights: ')
            print(output_layer_weights)  ## end for testing
        # end while
        print('In ' + str(epoch_number) + ' epochs, we obtained a validation accuracy of ' + str(
            current_accuracy) + ' and a training accuracy of '+str(test_accuracy)+' with a learning rate of ' + str(self.lr) + ' and a momentum of ' + str(
            self.momentum) + '.')
        print('Hidden layer weights: ')
        print(hidden_layer_weights)
        print('Output layer weights: ')
        print(output_layer_weights)
        return hidden_layer_weights, output_layer_weights
        #return self

    def predict(self, X,hidden_layer_weights=None,output_layer_weights=None):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        if hidden_layer_weights is None or output_layer_weights is None:
            return -999
        for x, target in zip(X, y):
            x = np.concatenate((x, [1]), axis=0)
            hidden_layer_net, output_layer_net, hidden_layer_z, output_layer_z, output_layer_z_binary = self.calculate_Z(
                hidden_layer_weights, x, output_layer_weights)
        return output_layer_z_binary
        pass

    def initialize_weights(self, weights=None, initial_weights=None):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        assert len(weights) == len(initial_weights)
        assert len(weights[0]) == len(initial_weights[0])
        return np.array(initial_weights, dtype=float)

    def score(self, X, y, hidden_layer_weights=None, output_layer_weights=None):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        incorrect = 0.
        if hidden_layer_weights is None or output_layer_weights is None:
            return -999
        for x, target in zip(X, y):
            x = np.concatenate((x, [1]), axis=0)
            hidden_layer_net, output_layer_net, hidden_layer_z, output_layer_z, output_layer_z_binary = self.calculate_Z(
                hidden_layer_weights, x, output_layer_weights)
            incorrect += np.abs(output_layer_z_binary-target).sum()
        #print("incorrect: "+str(incorrect))
        #print(len(y)*len(y[0]))
        return 1-(incorrect / (len(y)*len(y[0])))

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        return X, y
        pass

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    ## I Don't Know how you expect us to get weights without adding class variables or a required input
    def get_weights(self,weight=None):
        if weight is None:
            return -999
        return weight['hidden'], weight['output']
        pass

    def fnet(self, net):
        Zj = 1 / (1 + np.exp(-net))
        if np.isnan(Zj):
            Zj=.5
        assert not np.isnan(Zj)
        assert np.isfinite(Zj)
        return Zj
        pass

    def fprime(self, net):
        Z = self.fnet(net)
        return Z * (1 - Z)
        pass

    def output_delta(self, T, z, net):
        return (T - z) * self.fprime(net)
        pass

    def calculate_Z(self, hidden_layer_weights, x, output_layer_weights):
        hidden_layer_net = np.zeros(len(hidden_layer_weights))
        output_layer_net = np.zeros(len(output_layer_weights))
        hidden_layer_z = np.zeros(len(hidden_layer_weights))
        output_layer_z = np.zeros(len(output_layer_weights))
        output_layer_z_binary = np.zeros(len(output_layer_weights))

        for i in range(len(hidden_layer_weights)):
            hidden_layer_net[i] = np.dot(x, hidden_layer_weights[i])
            hidden_layer_z[i] = self.fnet(hidden_layer_net[i])
        for i in range(len(output_layer_z)):
            output_layer_net[i] = np.dot(hidden_layer_z, output_layer_weights[i][:-1])
            output_layer_net[i] += output_layer_weights[i][-1]
            output_layer_z[i] = self.fnet(output_layer_net[i])
        # for z, net, weights, input in zip(output_layer_z, output_layer_net, output_layer_weights, hidden_layer_z):
        #     net = np.dot(input, weights[:-1])
        #     net += weights[-1]
        #     z = self.fnet(net)
        output_layer_z_binary = np.around(output_layer_z)

        return hidden_layer_net, output_layer_net, hidden_layer_z, output_layer_z, output_layer_z_binary

    def calculate_delta(self, hidden_layer_z, output_layer_z, hidden_layer_net, output_layer_net, output_layer_weights,
                        y):

        output_layer_delta = np.zeros(len(output_layer_z))
        hidden_layer_delta = np.zeros(len(hidden_layer_z))
        # calculate output layer delta
        for i in range(len(y)):
            output_layer_delta[i] = self.output_delta(y[i], output_layer_z[i], output_layer_net[i])
        # for target, z, net, delta in zip(y, output_layer_z, output_layer_net, output_layer_delta):
        #    delta = self.output_delta(target, z, net)

        # calculate hidden layer delta
        ## debugged to here
        for j in range(len(hidden_layer_delta)):
            sum = 0
            for k in range(len(output_layer_weights)):
                sum += output_layer_delta[k] * output_layer_weights[k][j]
            hidden_layer_delta[j] = sum * self.fprime(hidden_layer_net[j])
        # for delta, weights, net in zip(hidden_layer_delta, output_layer_weights, hidden_layer_net):
        #    sum = 0
        #    for w in weights:
        #        sum += delta * w
        #    delta = sum * self.fprime(net)
        return hidden_layer_delta, output_layer_delta

    def calculate_delta_weights(self, hidden_layer_z, output_layer_z, hidden_layer_net, output_layer_net,
                                hidden_layer_weights, output_layer_weights, x, y, hidden_layer_old_delta_weights,
                                output_layer_old_delta_weights):

        hidden_layer_delta_weights = np.array(hidden_layer_weights, copy=True, dtype=float)
        output_layer_delta_weights = np.array(output_layer_weights, copy=True, dtype=float)

        hidden_layer_delta, output_layer_delta = self.calculate_delta(hidden_layer_z, output_layer_z, hidden_layer_net,
                                                                      output_layer_net, output_layer_weights, y, )

        for i in range(len(output_layer_delta_weights)):
            for j in range(output_layer_delta_weights.shape[1]-1):
                without_momentum =self.lr * output_layer_delta[i] * hidden_layer_z[j]
                momentum = self.momentum * output_layer_old_delta_weights[i][j]
                output_layer_delta_weights[i, j] = without_momentum + momentum
                #print(momentum)
            output_layer_delta_weights[i,-1] = self.lr * output_layer_delta[i] * 1 + self.momentum * output_layer_old_delta_weights[i][-1]
        # for delta_weights, delta, old_delta_weights in zip(output_layer_delta_weights, output_layer_delta,
        #                                                   output_layer_old_delta_weights):
        #    for delta_weight, z, old_delta_weight in zip(delta_weights, hidden_layer_z, old_delta_weights):
        #        delta_weight = self.lr * delta * z + self.momentum * old_delta_weight
        for i in range(len(hidden_layer_delta_weights)):
            for j in range((len(hidden_layer_delta_weights[i]))):
                hidden_layer_delta_weights[i][j] = self.lr * hidden_layer_delta[i] * x[j] \
                                                   + self.momentum * hidden_layer_old_delta_weights[i][j]
        # for delta_weights, delta, old_delta_weights in zip(hidden_layer_delta_weights, hidden_layer_delta,
        #                                                   hidden_layer_old_delta_weights):
        #    for delta_weight, z, old_delta_weight in zip(delta_weights, x, old_delta_weights):  ## only current row of X
        #        delta_weight = self.lr * delta * z + self.momentum * old_delta_weight

        return hidden_layer_delta_weights, output_layer_delta_weights,
