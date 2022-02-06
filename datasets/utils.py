import matplotlib.pyplot as plt
import numpy as np

from datasets.shapes import Shapes
from utils.value_checking import check_boundary_value


def dataset_from_config(config) -> Shapes:
    # parse dataset parameters
    dataset_params: dict = config.get("dataset", {})
    split_ratio = dataset_params.get("split_ratio", [0.7, 0.2, 0.1])
    img_dims = check_boundary_value("img_dims", dataset_params.get("img_dims", 20), min_val=10, max_val=50)
    width_range = dataset_params.get("width_range", [10, 15])
    height_range = dataset_params.get("height_range", [10, 15])
    img_noise = check_boundary_value("img_noise", dataset_params.get("img_noise", 0.02), min_val=0, max_val=1)
    flatten = dataset_params.get("flatten", False)
    size = check_boundary_value("size", dataset_params.get("size", 1000), min_val=0)
    centred = dataset_params.get("centred", False)
    normalise = dataset_params.get("normalise", False)

    dataset = Shapes(
        split_ratio=split_ratio,
        img_dims=img_dims,
        width_range=width_range,
        height_range=height_range,
        img_noise=img_noise,
        dataset_size=size,
        flatten=flatten,
        centred=centred,
        normalise=normalise
    )

    return dataset


def plot_images(imgs, labels=None):
    for i, img in enumerate(imgs):
        if len(img.shape) == 1:
            img = img.reshape((int(np.sqrt(img.shape[0])), int(np.sqrt(img.shape[0]))))
        plt.imshow(img, cmap='gray')
        if labels:
            plt.title(labels[i])
        plt.show()
