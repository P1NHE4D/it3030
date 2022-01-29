import numpy as np
from nn.loss_functions import LossFunction
from tqdm import tqdm
import matplotlib.pyplot as plt


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

    def fit(self, x_train, y_train, x_val, y_val):

        progress = tqdm(range(1, self.epochs + 1))
        train_epoch_loss = []
        val_epoch_loss = []
        for epoch in progress:
            train_loss = []
            val_loss = []
            for x, y in zip(x_train, y_train):
                forward_val = x
                # forward pass
                for layer in self.layers:
                    forward_val = layer.forward(forward_val)

                # loss
                loss = self.loss_function.loss(forward_val, y)
                train_loss.append(loss)

                # backprop
                backwards_val = self.loss_function.gradient(forward_val, y)
                for layer in reversed(self.layers):
                    backwards_val = layer.backward(backwards_val)

            # update weights
            for layer in self.layers:
                layer.update_weights(self.learning_rate)

            for x, y in zip(x_val, y_val):
                forward_val = x
                # forward pass
                for layer in self.layers:
                    forward_val = layer.forward(forward_val)

                # loss
                loss = self.loss_function.loss(forward_val, y)
                val_loss.append(loss)

            train_epoch_loss.append(np.mean(train_loss))
            val_epoch_loss.append(np.mean(val_loss))

            progress.set_description(
                "Epoch: {}".format(epoch) +
                " | "
                "Avg. train loss: {}".format(np.mean(train_loss)) +
                " | " +
                "Avg. val loss: {}".format(np.mean(val_loss))
            )

        plt.plot(range(1, self.epochs + 1), train_epoch_loss, label='Train')
        plt.plot(range(1, self.epochs + 1), val_epoch_loss, label='Val')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def predict(self, X):
        res = []
        for x in X:
            forward_val = x
            for layer in self.layers:
                forward_val = layer.forward(forward_val)
            res.append(forward_val)
        return np.array(res)
