import numpy as np
from nn.loss_functions import CrossEntropy
from tqdm import tqdm
import matplotlib.pyplot as plt


class SequentialNetwork:

    def __init__(self, learning_rate=0.001, epochs=100, loss_function=CrossEntropy(), visualize=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_function = loss_function
        self.visualize = visualize
        self.layers = []

    def add(self, layer):
        """
        Adds given layer to network
        :param layer: layer to be added
        """
        self.layers.append(layer)

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        """
        approximates a mapping function based on the given training data

        :param x_train: training set
        :param y_train: target values of the training set
        :param x_val: validation set
        :param y_val: target values of the validation set
        """

        progress = tqdm(range(1, self.epochs + 1))
        train_epoch_loss = []
        val_epoch_loss = []

        ###### training ######
        for epoch in progress:
            train_loss = []
            val_loss = []
            progress_desc = "Epoch: {}".format(epoch)

            # compute loss and gradient for each data instance in given batch
            for x, y in zip(x_train, y_train):

                # forward pass
                forward_val = x
                for layer in self.layers:
                    forward_val = layer.forward(forward_val)

                # loss and regularization loss (if enabled)
                loss = self.loss_function.loss(forward_val, y) + np.sum(
                    [layer.layer_penalty() for layer in self.layers])
                train_loss.append(loss)

                # backpropagation
                backwards_val = self.loss_function.gradient(forward_val, y)
                for layer in reversed(self.layers):
                    backwards_val = layer.backward(backwards_val)

            # update weights based on accumulated gradient
            for layer in self.layers:
                layer.update_weights(self.learning_rate)

            # compute average training loss
            train_epoch_loss.append(np.mean(train_loss))
            progress_desc += " | Avg. train loss: {}".format(np.mean(train_loss))

            if x_val is not None and y_val is not None:
                # compute loss for validation set
                for x, y in zip(x_val, y_val):

                    # forward pass
                    forward_val = x
                    for layer in self.layers:
                        forward_val = layer.forward(forward_val)

                    # loss
                    loss = self.loss_function.loss(forward_val, y)
                    val_loss.append(loss)

                # compute average validation loss
                val_epoch_loss.append(np.mean(val_loss))
                progress_desc += " | Avg. val loss: {}".format(np.mean(val_loss))

            progress.set_description(progress_desc)

        # visualise training and validation loss
        if self.visualize:
            plt.plot(range(1, self.epochs + 1), train_epoch_loss, label='Train')
            plt.plot(range(1, self.epochs + 1), val_epoch_loss, label='Val')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

    def predict(self, X):
        """
        predicts the values for given dataset

        :param X: dataset for which the labels should be predicted
        :return: predicted labels
        """
        res = []
        for x in X:
            forward_val = x
            for layer in self.layers:
                forward_val = layer.forward(forward_val)
            res.append(forward_val)
        return np.array(res)
