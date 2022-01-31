import json
from datasets.utils import dataset_from_config
from nn.loss_functions import CrossEntropy
import numpy as np

from nn.utils import network_from_config


def main():
    with open("config.json") as f:
        config = json.load(f)
    model = network_from_config(config)
    dataset = dataset_from_config(config)
    x_train, y_train, x_val, y_val, x_test, y_test = dataset.generate_dataset()

    model.fit(x_train, y_train, x_val, y_val)

    y_test_pred = model.predict(x_test)
    test_loss = []
    for gt, pred in zip(y_test, y_test_pred):
        loss = CrossEntropy().loss(y_pred=pred, y_true=gt)
        test_loss.append(loss)
    print("Test avg. loss: {:.2f}".format(np.mean(test_loss)))

    # model = SequentialNetwork(config)
    # model.add(Flatten())
    # model.add(Dense(units=64, activation=Relu(), regularizer=L2(0.001)))
    # model.add(Dense(units=32, activation=Relu(), regularizer=L2(0.001)))
    # model.add(Dense(units=4, activation=Linear(), regularizer=L2(0.001)))
    # model.add(Softmax())
    # model.compile(loss_function=CrossEntropy())
#
    # s = Shapes(config)
    # x_train, y_train, x_val, y_val, x_test, y_test = s.generate_dataset()
    # # TODO: normalize data to improve results
#
    # model.fit(x_train, y_train, x_val, y_val)
    # y_test_pred = model.predict(x_test)
    # test_loss = []
    # for gt, pred in zip(y_test, y_test_pred):
    #     loss = CrossEntropy().loss(y_pred=pred, y_true=gt)
    #     test_loss.append(loss)
    # print("Test avg. loss: {:.2f}".format(np.mean(test_loss)))


if __name__ == '__main__':
    main()
