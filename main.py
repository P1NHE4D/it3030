import json

from nn.core import SequentialNetwork
from nn.activation_functions import Relu, Linear

from nn.layers import Dense, Flatten, Softmax
from nn.loss_functions import CrossEntropy


def main():
    with open("config.json") as f:
        config = json.load(f)
    model = SequentialNetwork(config)
    model.add(Flatten())
    model.add(Dense(units=128, activation=Relu()))
    model.add(Dense(units=128, activation=Relu()))
    model.add(Dense(units=10, activation=Linear()))
    model.add(Softmax())
    model.compile(loss_function=CrossEntropy())


if __name__ == '__main__':
    main()
