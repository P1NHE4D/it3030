import json

from nn.core import SequentialNetwork
from nn.activation_functions import Relu, Sigmoid
import pandas as pd
import numpy as np
import tensorflow as tf

from nn.layers import Dense, Flatten, Softmax
from nn.loss_functions import CrossEntropy


def main():
    with open("config.json") as f:
        config = json.load(f)
    model = SequentialNetwork(config)
    model.add(Flatten())
    model.add(Dense(units=2, activation=Relu()))
    model.add(Dense(units=10, activation=Relu()))
    model.add(Dense(units=2, activation=Relu()))
    model.add(Softmax())
    model.compile(loss_function=CrossEntropy())

    x_train = pd.read_csv("data/IRISdata.csv", usecols=["SepalWidthCm", "PetalWidthCm"]).to_numpy()
    y_train = pd.read_csv("data/IRIStargets.csv", usecols=["Species"])
    y_train[y_train.Species == -1] = 0
    y_train = y_train.to_numpy()

    # normalise data
    #x_train = tf.keras.utils.normalize(x_train, axis=1)
    y_train = tf.one_hot(y_train, 2)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    print(y_train, y_pred)

if __name__ == '__main__':
    main()
