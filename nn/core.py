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
        progress = tqdm(range(self.epochs))
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


class DenseLayer:
    """
    each layer comprises a set of weights and biases associated with the neurons that feed into the layer
    should be able to define the input size and output size
    """

    def __init__(self, units, activation: ActivationFunction):
        self.units = units
        self.activation = activation
        self.weights = None  # weights should be n + 1 to include the bias
        self.input = None
        self.output_sum = None
        self.weight_update = None

    def forward(self, X: ndarray):
        """
        :return: activation value based on the input
        """
        # add column with ones to handle the bias
        # X = np.hstack([X, [1]])

        # init weights if they have not been defined yet
        if self.weights is None:
            # self.weights = np.random.random((len(X), self.units))
            self.weights = np.full((self.units, len(X)), 0.1)
            self.weight_update = np.zeros((self.units, len(X)))

        # store data for backpropagation
        self.input = X
        self.output_sum = np.dot(self.weights, X)

        # compute activation value
        return self.activation.function(self.output_sum)

    def backward(self, J_L_Z: ndarray):
        """

        :param j_sum: gradient information passed in from succeeding layer
        :return: gradient with respect to preceeding layer
        """
        r = np.diag(self.activation.gradient(self.output_sum)) # 2 x 2
        J_Z_Y = np.dot(r, self.weights) # 2 x 2 dot 2 x 32 -> 2 x 32
        J_Z_W = np.outer(self.input, self.activation.gradient(self.output_sum)) # outer(32, 2) -> 2 x 32
        J_L_W = J_L_Z * J_Z_W #
        J_L_Y = np.dot(J_L_Z, J_Z_Y).reshape((-1, ))


        self.weight_update += J_L_W.T
        return J_L_Y
