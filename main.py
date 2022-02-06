import json
from datasets.utils import dataset_from_config, plot_images
from nn.loss_functions import CrossEntropy
import numpy as np
from nn.utils import network_from_config


def main():
    with open("config.json") as f:
        config = json.load(f)

    # create neural network based on config
    model = network_from_config(config)

    # generate dataset
    dataset = dataset_from_config(config)
    x_train, y_train, x_val, y_val, x_test, y_test = dataset.generate_dataset()

    # display images if enabled
    image_viewer = config.get("image_viewer", {})
    if image_viewer.get("enabled", False):
        img_set = image_viewer.get("set", "train")
        img_set = x_train if img_set == "train" else x_val if img_set == "val" else x_test
        plot_count = image_viewer.get("plot_count", 10)
        if image_viewer.get("random", False):
            idx = np.random.choice(range(0, len(img_set)), plot_count, replace=False)
        else:
            idx = np.arange(0, plot_count)
        plot_images(img_set[idx])

    # fit the model based on the training set and compute the loss of the validation set for each batch
    model.fit(x_train, y_train, x_val, y_val)

    # predict labels for test set
    y_test_pred = model.predict(x_test)

    # compute loss of the test set
    test_loss = []
    for gt, pred in zip(y_test, y_test_pred):
        loss = CrossEntropy().loss(y_pred=pred, y_true=gt)
        test_loss.append(loss)
    print("Test avg. loss: {:.2f}".format(np.mean(test_loss)))

    # display images with predicted labels if enabled
    if image_viewer.get("enabled", False):
        plot_count = image_viewer.get("plot_count", 10)
        if image_viewer.get("random", False):
            idxs = np.random.choice(range(0, len(x_test)), plot_count, replace=False)
        else:
            idxs = np.arange(0, plot_count)
        labels = []
        for idx in idxs:
            pred_idx = np.argmax(y_test_pred[idx])
            pred_lbl = "rectangle" if pred_idx == 0 else "cross" if pred_idx == 1 else "circle" if pred_idx == 2 else "triangle"
            labels.append("Prediction: {} Likelihood: {:.2f}".format(pred_lbl, y_test_pred[idx][pred_idx]))
        plot_images(x_test[idxs], labels)


if __name__ == '__main__':
    main()
