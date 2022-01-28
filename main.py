import json

from nn.core import SequentialNetwork
from nn.activation_functions import Relu, Sigmoid, Linear, Softmax
import pandas as pd
import numpy as np
import tensorflow as tf

from nn.layers import Dense, Flatten
from nn.loss_functions import CrossEntropy


def main():
    with open("config.json") as f:
        config = json.load(f)
    model = SequentialNetwork(config)
    model.add(Flatten())
    model.add(Dense(units=128, activation=Relu()))
    model.add(Dense(units=64, activation=Relu()))
    model.add(Dense(units=32, activation=Relu()))
    model.add(Dense(units=10, activation=Softmax()))
    model.compile(loss_function=CrossEntropy())

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(y_train)

    idx = np.random.choice(np.arange(len(x_train)), 500, replace=False)
    x_train = x_train[idx]
    y_train = y_train[idx]

    # normalise data
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    y_train = tf.one_hot(y_train, 10)
    print(y_train)
    model.fit(x_train, y_train)


if __name__ == '__main__':
    main()
