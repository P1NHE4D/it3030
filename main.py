import json

from nn.core import SequentialNetwork, Layer
from nn.activation_functions import Relu, Softmax, Sigmoid, Linear
import pandas as pd
import numpy as np


def main():
    with open("config.json") as f:
        config = json.load(f)
    model = SequentialNetwork(config)
    model.add(Layer(units=64, activation=Sigmoid()))
    model.add(Layer(units=32, activation=Sigmoid()))
    model.add(Layer(units=1, activation=Sigmoid()))
    model.compile()


if __name__ == '__main__':
    main()
