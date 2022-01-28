from abc import ABC, abstractmethod
import numpy as np


class ActivationFunction(ABC):

    @abstractmethod
    def function(self, X):
        pass

    @abstractmethod
    def gradient(self, X):
        pass


class Sigmoid(ActivationFunction):
    def function(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return x * (1 - x)


class Tanh(ActivationFunction):

    def function(self, x):
        return np.tanh(x)

    def gradient(self, x):
        return 1 - np.square(np.tanh(x))


class Relu(ActivationFunction):

    def function(self, x):
        x[x <= 0] = 0
        return x

    def gradient(self, x):
        x[x > 0] = 1
        x[x <= 0] = 0
        return x


class Softmax(ActivationFunction):

    def function(self, x):
        return np.exp(x) / np.exp(x).sum()

    def gradient(self, x):
        return np.diag(x) - np.outer(x, x)
