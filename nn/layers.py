from abc import abstractmethod, ABC

from numpy import ndarray

from nn.activation_functions import ActivationFunction, Softmax
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

        self.input = X
        # store data for backpropagation
        self.output_sum = np.dot(X, self.weights)

        # compute activation value
        return self.activation.function(self.output_sum)

    def backward(self, J_L_Z: ndarray):
        # TODO
        if isinstance(self.activation, Softmax):
            J_S = self.activation.gradient(self.output_sum)
            J_L_Z = np.dot(J_L_Z, J_S)
            J_Z_sum = np.diag(self.output_sum)
        else:
            J_Z_sum = np.diag(self.activation.gradient(self.output_sum))
        J_Z_Y = np.dot(J_Z_sum, self.weights[:-1].T)
        J_Z_W = np.outer(self.input, J_Z_sum.diagonal())
        J_L_Y = np.dot(J_L_Z, J_Z_Y)
        J_L_W = J_L_Z * J_Z_W

        self.weight_update += J_L_W
        return J_L_Y

    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.weight_update
        self.weight_update = np.zeros(self.weight_update.shape)


class Flatten(Layer):

    def forward(self, X):
        return X.reshape(-1)

    def backward(self, J_L_Z: ndarray):
        return J_L_Z

    def update_weights(self, learning_rate):
        pass
