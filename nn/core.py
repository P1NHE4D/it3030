from numpy.core.records import ndarray
import numpy as np
from nn.activation_functions import ActivationFunction
from nn.loss_functions import MSE
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

    def compile(self, loss_function: ActivationFunction = MSE()):
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
                layer.weights -= self.learning_rate * layer.weight_update
                layer.weight_update = np.zeros(layer.weight_update.shape)

            progress.set_description(
                "Epoch: {}".format(epoch) +
                " Loss: {}".format(loss)
            )
        for layer in self.layers:
            print(layer.weights)

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


class Layer:
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
            # self.weights = np.full((len(X), self.units), 0.1)
            # self.bias = np.full((len(X)), 0.1)
            self.weight_update = np.zeros((len(X), self.units))

        self.input = X
        # store data for backpropagation
        self.output_sum = np.dot(X, self.weights)

        # compute activation value
        return self.activation.function(self.output_sum)

    def backward(self, J_L_Z: ndarray):
        """

        :param j_sum: gradient information passed in from succeeding layer
        :return: gradient with respect to preceeding layer
        """
        J_Z_sum = np.diag(self.activation.gradient(self.output_sum))
        J_Z_Y = np.dot(J_Z_sum, self.weights[:-1].T)
        J_Z_W = np.outer(self.input, J_Z_sum.diagonal())
        J_L_Y = np.dot(J_L_Z, J_Z_Y)
        J_L_W = J_L_Z * J_Z_W

        self.weight_update += J_L_W
        return J_L_Y
