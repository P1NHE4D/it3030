import json

from nn.core import SequentialNetwork
from nn.activation_functions import Relu, Linear

from nn.layers import Dense, Flatten, Softmax
from nn.loss_functions import CrossEntropy
from datasets.shapes import Shapes
import numpy as np
import matplotlib.pyplot as plt


def main():
    with open("config.json") as f:
        config = json.load(f)
    model = SequentialNetwork(config)
    model.add(Flatten())
    model.add(Dense(units=64, activation=Relu()))
    model.add(Dense(units=32, activation=Relu()))
    model.add(Dense(units=4, activation=Linear()))
    model.add(Softmax())
    model.compile(loss_function=CrossEntropy())

    s = Shapes(config)
    x_train, y_train, x_val, y_val, x_test, y_test = s.generate_dataset()

    model.fit(x_train, y_train, x_val, y_val)
    y_test_pred = model.predict(x_test)
    test_loss = []
    for gt, pred in zip(y_test, y_test_pred):
        loss = CrossEntropy().loss(y_pred=pred, y_true=gt)
        test_loss.append(loss)
    print("Test avg. loss: {:.2f}".format(np.mean(test_loss)))


if __name__ == '__main__':
    main()
