import json

from nn.core import SequentialNetwork
from nn.activation_functions import Relu, Linear

from nn.layers import Dense, Flatten, Softmax
from nn.loss_functions import CrossEntropy
from datasets.shapes import Shapes
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
    plt.imshow(x_train[0], cmap='gray')
    plt.show()

    model.fit(x_train, y_train, x_val, y_val)
    y_pred = model.predict(x_val[0:5])
    for y_t, y in zip(y_val[0:5], y_pred):
        print("GT: Rectangle: {} | Cross: {} | Circle: {} | Triangle: {}".format(y_t[0], y_t[1], y_t[2], y_t[3]))
        print("Prediction: Rectangle: {:.2f} | Cross: {:.2f} | Circle: {:.2f} | Triangle: {:.2f}".format(y[0], y[1], y[2], y[3]))


if __name__ == '__main__':
    main()
