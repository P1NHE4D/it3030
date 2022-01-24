from abc import ABC, abstractmethod
import numpy as np


class ActivationFunction(ABC):

    @abstractmethod
    def function_value(self, X):
        pass

    @abstractmethod
    def gradient(self, X):
        pass


class Sigmoid(ActivationFunction):
    def function_value(self, X):
        f = np.vectorize(lambda x: 1 / (1 + np.e ** -x))
        return f(X)

    def gradient(self, X):
        f = np.vectorize(lambda x: self.function_value(x) * (1 - self.function_value(x)))
        return f(X)


class Tanh(ActivationFunction):

    def function_value(self, X):
        return np.tanh(X)

    def gradient(self, X):
        f = np.vectorize(lambda x: 1 - np.tanh(x) ** 2)
        return f(X)


class Relu(ActivationFunction):

    def function_value(self, X):
        return np.max(X, 0)

    def gradient(self, X):
        v = np.copy(X)
        v[v > 0] = 1
        v[v < 0] = 0
        if len(v[v == 0]) != 0:
            raise Exception("Gradient of ReLu function undefined at x = 0!")
        return v


class Linear(ActivationFunction):
    def function_value(self, X):
        return X

    def gradient(self, X):
        return 1


class Softmax(ActivationFunction):

    def function_value(self, X):
        pass

    def gradient(self, X):
        pass


class MSE(ActivationFunction):

    def function_value(self, X):
        pass

    def gradient(self, X):
        pass


class CrossEntropy(ActivationFunction):

    def function_value(self, X):
        pass

    def gradient(self, X):
        pass
