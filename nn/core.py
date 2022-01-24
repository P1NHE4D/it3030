from numpy.core.records import ndarray

from nn.functions import ActivationFunction, MSE


class SequentialNetwork:
    """
    then call fit with the train data and train labels
    """
    def __init__(self, config):
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]

    def add(self, layer):
        """
        Adds given layer to network
        :param layer: layer to be added
        """
        pass

    def compile(self, loss_function: ActivationFunction = MSE()):
        # initialise all the weights etc. - setup work
        pass

    def fit(self, X, y):
        # for batch in batches:
        # for each layer in network
        # conduct forward pass of X
        # compute loss and output some information, e.g. accuracy
        # conduct backward pass
        pass

    def predict(self, X):
        # for each layer in network
        # conduct forward pass of X
        # return output of final layer
        pass


class DenseLayer:
    """
    each layer comprises a set of weights and biases associated with the neurons that feed into the layer
    should be able to define the input size and output size
    """

    def __init__(self, units, activation):
        self.units = units
        self.activation = activation
        self.weights = None

    def forward(self, X: ndarray):
        """
        :return: activation value based on the input
        """
        return self.activation(X.dot(self.weights))

    def backward(self, dX):
        """

        :param dX:
        :return:
        """
        # compute gradient using passed gradient information
        # update weights using gradient information
        # pass on gradient information to preceeding layer
        pass
