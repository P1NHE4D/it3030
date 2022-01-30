from abc import abstractmethod, ABC
from numpy import ndarray
from nn.activation_functions import ActivationFunction
import numpy as np

from nn.regularizers import Regularizer


class Layer(ABC):

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, J_L_Z: ndarray):
        pass

    def update_weights(self, learning_rate):
        pass

    def layer_penalty(self):
        return 0


class Dense(Layer):

    def __init__(self, units, activation: ActivationFunction, regularizer: Regularizer = None):
        self.units = units
        self.activation = activation
        self.regularizer = regularizer
        self.weights = None
        self.input = None
        self.weight_update = None
        self.output = None

    def forward(self, X: ndarray):
        # add 1 for bias
        X = np.hstack([X, [1]])

        # init weights if they have not been defined yet
        if self.weights is None:
            self.weights = np.random.uniform(-0.1, 0.1, (len(X), self.units))
            self.weight_update = np.zeros((len(X), self.units))

        # store data for backpropagation
        self.input = X
        output_sum = np.dot(X, self.weights)
        self.output = self.activation.function(output_sum)

        # compute activation value
        return self.output

    def backward(self, J_L_N: ndarray):
        J_N_SUM = np.diag(self.activation.gradient(self.output))
        J_N_M = np.dot(J_N_SUM, self.weights[:-1].T)
        J_N_W = np.outer(self.input, J_N_SUM.diagonal())
        J_L_M = np.dot(J_L_N, J_N_M)
        J_L_W = J_L_N * J_N_W
        if self.regularizer:
            J_L_W += self.regularizer.factor * self.regularizer.gradient(self.weights)

        self.weight_update += J_L_W
        return J_L_M

    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.weight_update
        self.weight_update = np.zeros(self.weight_update.shape)

    def layer_penalty(self):
        if self.regularizer:
            return self.regularizer.factor * self.regularizer.penalty(self.weights)
        return 0


class Softmax(Layer):

    def __init__(self):
        self.output = None

    def forward(self, X):
        self.output = np.exp(X) / np.exp(X).sum()
        return self.output

    def backward(self, J_L_Z: ndarray):
        J_S = np.diag(self.output) - np.outer(self.output, self.output)
        return np.dot(J_L_Z, J_S)


class Flatten(Layer):

    def forward(self, X):
        return X.reshape(-1)

    def backward(self, J_L_Z: ndarray):
        return J_L_Z
