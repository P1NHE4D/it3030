from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):

    @abstractmethod
    def loss(self, y_pred, y_true):
        pass

    @abstractmethod
    def gradient(self, y_pred, y_true):
        pass


class MSE(LossFunction):
    def loss(self, y_pred, y_true):
        loss = np.mean(np.square(y_pred - y_true))
        return loss

    def gradient(self, y_pred, y_true):
        gradient = 2 * np.mean((y_pred - y_true))
        return gradient


class CrossEntropy(LossFunction):
    def loss(self, y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred))

    def gradient(self, y_pred, y_true):
        return y_pred - y_true
