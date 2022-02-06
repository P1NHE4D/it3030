from abc import abstractmethod, ABC
from numpy import ndarray
from nn.activation_functions import ActivationFunction
import numpy as np

from nn.regularizers import Regularizer


class Layer(ABC):

    @abstractmethod
    def forward(self, input_data):
        """
        Performs a forward pass through the network with the given input data
        :param input_data: input data (1xN-vector)
        """
        pass

    @abstractmethod
    def backward(self, J_L_N: ndarray):
        """
        Computes the gradient of the network based on the gradient of the network loss

        :param J_L_N: gradient of succeeding layer with respect to this layer
        :return: gradient of this layer with respect to previous layer
        """
        pass

    def update_weights(self, learning_rate):
        """
        Updates the weights using a fraction of the weight update based on the learning rate
        :param learning_rate: learning rate
        """
        pass

    def layer_penalty(self):
        """
        Returns the weight penalty based on the regularization function
        Returns 0 if no regularization function is defined

        :return: penalty or 0
        """
        return 0


class Dense(Layer):

    def __init__(self, units, activation: ActivationFunction, wr: list, regularizer: Regularizer = None):
        self.units = units
        self.activation = activation
        self.regularizer = regularizer
        self.wr = wr
        self.weights = None
        self.input = None
        self.weight_update = None
        self.output = None

    def forward(self, input_data: ndarray):

        # add 1 for bias
        input_data = np.hstack([input_data, [1]])

        # init weights if they have not yet been defined
        if self.weights is None:
            self.weights = np.random.uniform(self.wr[0], self.wr[1], (len(input_data), self.units))
            self.weight_update = np.zeros((len(input_data), self.units))

        # store data for backpropagation
        self.input = input_data
        output_sum = np.dot(input_data, self.weights)
        self.output = self.activation.function(output_sum)

        # compute activation value
        return self.output

    def backward(self, J_L_N: ndarray):
        # gradient with respect to the activation value of this layer
        J_N_SUM = np.diag(self.activation.gradient(self.output))

        # gradients from the current layer with respect to the previous layer
        J_N_M = np.dot(J_N_SUM, self.weights[:-1].T)

        # gradients from current layer with respect to the incoming weights
        J_N_W = np.outer(self.input, J_N_SUM.diagonal())

        # gradients from loss with respect to the previous layer
        J_L_M = np.dot(J_L_N, J_N_M)

        # gradients from loss with respect to the incoming weights
        J_L_W = J_L_N * J_N_W

        # adding regularization penalty if enabled
        if self.regularizer:
            J_L_W += self.regularizer.factor * self.regularizer.gradient(self.weights)

        # adding weight gradient to previous weight gradients of the current batch
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

    def forward(self, input_data):
        self.output = np.exp(input_data) / np.exp(input_data).sum()
        return self.output

    def backward(self, J_L_N: ndarray):
        J_S = np.diag(self.output) - np.outer(self.output, self.output)
        return np.dot(J_L_N, J_S)


class Flatten(Layer):

    def forward(self, input_data):
        return input_data.reshape(-1)

    def backward(self, J_L_N: ndarray):
        return J_L_N
