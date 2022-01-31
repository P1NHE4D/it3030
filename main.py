import json

import matplotlib.pyplot as plt

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

    image_viewer = config.get("image_viewer", {})
    if image_viewer.get("enabled", False):
        img_set = image_viewer.get("set", "train")
        img_set = x_train if img_set == "train" else x_val if img_set == "val" else x_test
        plot_count = image_viewer.get("plot_count", 10)
        idx = np.random.choice(range(0, len(img_set)), plot_count, replace=False)
        for i in idx:
            plt.imshow(img_set[i], cmap="gray")
            plt.show()

    model.fit(x_train, y_train, x_val, y_val)
    y_test_pred = model.predict(x_test)
    test_loss = []
    for gt, pred in zip(y_test, y_test_pred):
        loss = CrossEntropy().loss(y_pred=pred, y_true=gt)
        test_loss.append(loss)
    print("Test avg. loss: {:.2f}".format(np.mean(test_loss)))

    if image_viewer.get("enabled", False) and image_viewer.get("plot_preds", False):
        plot_count = image_viewer.get("plot_count", 10)
        idxs = np.random.choice(range(0, len(x_test)), plot_count, replace=False)
        for idx in idxs:
            pred_idx = np.argmax(y_test_pred[idx])
            pred_lbl = "rectangle" if pred_idx == 0 else "cross" if pred_idx == 1 else "circle" if pred_idx == 2 else "triangle"
            plt.imshow(x_test[idx], cmap="gray")
            plt.title("Prediction: {} Likelihood: {}".format(pred_lbl, y_test_pred[idx][pred_idx]))
            plt.show()


if __name__ == '__main__':
    main()
