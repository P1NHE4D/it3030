import json

from nn.core import SequentialNetwork, DenseLayer
from nn.activation_functions import Relu, Softmax, Sigmoid, Linear
import pandas as pd
import numpy as np


def main():
    with open("config.json") as f:
        config = json.load(f)
    model = SequentialNetwork(config)
    model.add(DenseLayer(units=2, activation=Sigmoid()))
    model.add(DenseLayer(units=1, activation=Sigmoid()))
    model.compile()
    X = np.array([[20, 5, 13]])
    y = np.array([23])
    model.fit(X, y)
    print(model.predict(X[0:5]))


if __name__ == '__main__':
    main()
