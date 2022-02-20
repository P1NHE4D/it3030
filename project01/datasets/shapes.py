import numpy as np
import cv2 as cv
from tqdm import tqdm


class Shapes:

    def __init__(
            self,
            split_ratio,
            img_dims,
            width_range,
            height_range,
            img_noise,
            dataset_size,
            normalise=True,
            flatten=False,
            centred=False,
    ):
        self.split_ratio = split_ratio
        self.img_dims = img_dims
        self.width_range = width_range
        self.height_range = height_range
        self.img_noise = img_noise
        self.dataset_size = dataset_size
        self.normalise = normalise
        self.flatten = flatten
        self.centred = centred

    def draw_rectangle(self, img, width, height, x, y):
        """
        Adds a rectangle to the image

        :param img: image
        :param width: width of the rectangle
        :param height: height of the rectangle
        :param x: x coordinate
        :param y: y coordinate
        :return: image with rectangle
        """

        img[y, x:x + width - 1] = 255
        img[y + height - 1, x:x + width] = 255
        img[y:y + height - 1, x] = 255
        img[y:y + height, x + width - 1] = 255
        return self.add_noise(img)

    def draw_cross(self, img, width, height, x, y):
        """
        Adds a cross to the image

        :param img: image
        :param width: width of the cross
        :param height: height of the cross
        :param x: x coordinate
        :param y: y coordinate
        :return: image with cross
        """

        img[y + round(((height - 1) / 2)), x:x + width] = 255
        img[y:y + height, x + round(((width - 1) / 2))] = 255
        return self.add_noise(img)

    def draw_circle(self, img):
        """
        Adds a circle to the image

        :param img: image
        :return img with circle
        """

        diameter = np.random.choice(np.arange(start=self.width_range[0], stop=self.width_range[1] + 1))
        radius = diameter // 2
        if self.centred:
            x = (self.img_dims // 2) - radius
            y = (self.img_dims // 2) - radius
        else:
            x = np.random.choice(np.arange(start=0, stop=img.shape[0] - diameter + 1))
            y = np.random.choice(np.arange(start=0, stop=img.shape[1] - diameter + 1))
        WHITE = (255, 255, 255)
        cv.circle(img, (y + radius, x + radius), radius, WHITE, 1)
        return self.add_noise(img)

    def draw_triangle(self, img, width, height, x, y):
        """
        Adds a triangle to the image

        :param img: image
        :param width: width of the triangle
        :param height: height of the triangle
        :param x: x coordinate
        :param y: y coordinate
        :return: image with triangle
        """

        c1 = (x, y)
        c2 = (x + width, y)
        c3 = (x + round((width / 2)), y + height)
        pts = [c1, c2, c3]
        WHITE = (255, 255, 255)
        cv.polylines(img, np.array([pts]), True, WHITE, 1)
        img = self.add_noise(img)
        rotation = np.random.choice(np.array([0, cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE]))
        if rotation != 0:
            img = cv.rotate(img, rotation)
        return img

    def add_noise(self, img):
        """
        Adds noise to a given image based on the noise frequency specified in the config
        :param img: image
        :return: image with noise
        """
        flat_img = img.reshape(-1)
        idx = np.random.choice(np.arange(start=0, stop=len(flat_img)), round(self.img_noise * len(flat_img)))
        flat_img[idx] = 255 - flat_img[idx]
        return flat_img.reshape(img.shape)

    def generate_dataset(self):
        """
        Generates a training, validation, and test set comprising images with 4 different shapes
        """

        images = []
        targets = []

        shape = "rectangle"
        progress = tqdm(range(self.dataset_size))
        progress.set_description("Images generated: ")
        for _ in progress:
            # generate base image
            img = np.zeros((self.img_dims, self.img_dims))

            # specify width and height of shape
            width = min(self.img_dims,
                        np.random.choice(np.arange(start=self.width_range[0], stop=self.width_range[1] + 1)))
            height = min(self.img_dims,
                         np.random.choice(np.arange(start=self.height_range[0], stop=self.height_range[1] + 1)))

            # specify coordinates of shape
            if self.centred:
                x = (self.img_dims // 2) - (width // 2)
                y = (self.img_dims // 2) - (height // 2)
            else:
                x = np.random.choice(np.arange(start=0, stop=img.shape[0] - width + 1))
                y = np.random.choice(np.arange(start=0, stop=img.shape[1] - height + 1))

            # add shape to image
            if shape == "rectangle":
                img = self.draw_rectangle(img=img, width=width, height=height, x=x, y=y)
                target = [1, 0, 0, 0]
                shape = "cross"
            elif shape == "cross":
                img = self.draw_cross(img=img, width=width, height=height, x=x, y=y)
                target = [0, 1, 0, 0]
                shape = "circle"
            elif shape == "circle":
                img = self.draw_circle(img=img)
                target = [0, 0, 1, 0]
                shape = "triangle"
            else:
                img = self.draw_triangle(img=img, width=width, height=height, x=x, y=y)
                target = [0, 0, 0, 1]
                shape = "rectangle"

            if self.normalise:
                img *= (1 / 255)

            if self.flatten:
                img = img.reshape(-1)

            images.append(img)
            targets.append(target)

        images = np.array(images)
        targets = np.array(targets)

        # allocate images randomly to training, validation, and test set
        train_size = round(self.split_ratio[0] * len(images))
        val_size = round(self.split_ratio[1] * len(images))
        idx = np.arange(start=0, stop=len(images))
        train_idx = np.random.choice(idx, size=train_size, replace=False)
        idx = np.setdiff1d(idx, train_idx)
        val_idx = np.random.choice(idx, size=val_size)
        test_idx = np.setdiff1d(idx, val_idx)

        x_train = images[train_idx]
        y_train = targets[train_idx]
        x_val = images[val_idx]
        y_val = targets[val_idx]
        x_test = images[test_idx]
        y_test = targets[test_idx]

        return x_train, y_train, x_val, y_val, x_test, y_test
