from abc import ABC, abstractmethod
import numpy as np


class Regularizer(ABC):

    def __init__(self, regularization_rate):
        self.factor = regularization_rate

    @abstractmethod
    def penalty(self, weights):
        pass

    @abstractmethod
    def gradient(self, weights):
        pass


class L1(Regularizer):

    def __init__(self, regularization_rate):
        super().__init__(regularization_rate)

    def penalty(self, weights):
        return 0.5 * np.sum(np.square(weights))

    def gradient(self, weights):
        return weights


class L2(Regularizer):

    def __init__(self, regularization_rate):
        super().__init__(regularization_rate)

    def penalty(self, weights):
        return np.sum(np.abs(weights))

    def gradient(self, weights):
        return np.sign(weights)
