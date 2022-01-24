import json

from nn.core import SequentialNetwork, DenseLayer
from nn.functions import Relu, Softmax


def main():
    with open("config.json") as f:
        config = json.load(f)
    model = SequentialNetwork(config)
    model.add(DenseLayer(units=128, activation=Relu()))
    model.compile(loss_function=Softmax())
    # model.fit()
    # model.predict()


if __name__ == '__main__':
    main()
