from abc import ABC, abstractmethod

from numpy.core.records import ndarray
import numpy as np
from nn.activation_functions import ActivationFunction
from nn.loss_functions import LossFunction
from tqdm import tqdm


class SequentialNetwork:
    """
    then call fit with the train data and train labels
    """

    def __init__(self, config):
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]
        self.layers = []
        self.loss_function = None

    def add(self, layer):
        """
        Adds given layer to network
        :param layer: layer to be added
        """
        self.layers.append(layer)

    def compile(self, loss_function: LossFunction):
        self.loss_function = loss_function

    def fit(self, X, Y):
        # for batch in batches:
        # for each layer in network
        #   conduct forward pass of X
        # compute loss based on output of last layer using loss function
        # compute accuracy etc. and print it to console
        # conduct backward pass
        progress = tqdm(range(1, self.epochs + 1))
        for epoch in progress:
            loss = None
            for x, y in zip(X, Y):
                forward_val = x
                # forward pass
                for layer in self.layers:
                    forward_val = layer.forward(forward_val)

                # loss
                loss = self.loss_function.loss(forward_val, y)

                # backprop
                backwards_val = self.loss_function.gradient(forward_val, y)
                for layer in reversed(self.layers):
                    backwards_val = layer.backward(backwards_val)

            # update weights
            for layer in self.layers:
                layer.update_weights(self.learning_rate)

            progress.set_description(
                "Epoch: {}".format(epoch) +
                " Loss: {}".format(loss)
            )

    def predict(self, X):
        # for each layer in network
        # conduct forward pass of X
        # return output of final layer
        res = []
        for x in X:
            forward_val = x
            for layer in self.layers:
                forward_val = layer.forward(forward_val)
            res.append(forward_val)
        return np.array(res)
