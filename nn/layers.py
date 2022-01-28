from abc import abstractmethod, ABC

from numpy import ndarray

from nn.activation_functions import ActivationFunction
import numpy as np


class Layer(ABC):

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, J_L_Z: ndarray):
        pass

    @abstractmethod
    def update_weights(self, learning_rate):
        pass


class Dense(Layer):
    """
    each layer comprises a set of weights and biases associated with the neurons that feed into the layer
    should be able to define the input size and output size
    """

    def __init__(self, units, activation: ActivationFunction):
        self.units = units
        self.activation = activation
        self.weights = None
        self.input = None
        self.output_sum = None
        self.weight_update = None

    def forward(self, X: ndarray):
        """
        :return: activation value based on the input
        """
        # add 1 for bias
        X = np.hstack([X, [1]])

        # init weights if they have not been defined yet
        if self.weights is None:
            self.weights = np.random.uniform(-0.1, 0.1, (len(X), self.units))
            self.weight_update = np.zeros((len(X), self.units))

        # store data for backpropagation
        self.input = X
        self.output_sum = np.dot(X, self.weights)

        # compute activation value
        return self.activation.function(self.output_sum)

    def backward(self, J_L_N: ndarray):
        J_N_SUM = np.diag(self.activation.gradient(self.output_sum))
        J_N_M = np.dot(J_N_SUM, self.weights[:-1].T)
        J_N_W = np.outer(self.input, J_N_SUM.diagonal())
        J_L_M = np.dot(J_L_N, J_N_M)
        J_L_W = J_L_N * J_N_W

        self.weight_update += J_L_W
        return J_L_M

    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.weight_update
        self.weight_update = np.zeros(self.weight_update.shape)


class Softmax(Layer):

    def __init__(self):
        self.input = None

    def forward(self, X):
        self.input = X
        return np.exp(X) / np.exp(X).sum()

    def backward(self, J_L_Z: ndarray):
        J_S = np.diag(self.input) - np.outer(self.input, self.input)
        return np.dot(J_L_Z, J_S)

    def update_weights(self, learning_rate):
        pass


class Flatten(Layer):

    def forward(self, X):
        return X.reshape(-1)

    def backward(self, J_L_Z: ndarray):
        return J_L_Z

    def update_weights(self, learning_rate):
        pass
