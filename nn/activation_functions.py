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
        v = np.copy(x)
        v[v < 0] = 0
        return v

    def gradient(self, x):
        v = np.copy(x)
        v[v > 0] = 1
        v[v < 0] = 0
        if len(v[v == 0]) != 0:
            raise Exception("Gradient of ReLu function undefined at x = 0!")
        return v


class Linear(ActivationFunction):
    def function(self, x):
        return x

    def gradient(self, x):
        return np.ones(x.shape)


class Softmax(ActivationFunction):

    def function(self, x):
        return np.exp(x) / np.exp(x).sum()

    def gradient(self, x):
        return np.diag(x) - np.outer(x, x)
