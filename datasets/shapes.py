import numpy as np


class Shapes:

    def __init__(self, config):
        self.split_ratio = config["split_ratio"]
        self.img_dims = config["img_dims"]
        if not 10 <= self.img_dims <= 50:
            raise Exception("Image dimensions must be between 10 and 50!")

        self.width_range = config["width_range"]
        self.height_range = config["height_range"]
        self.img_noise = config["img_noise"]
        self.flatten = config["flatten"]

    def draw_rectangle(self):
        img = np.zeros((self.img_dims, self.img_dims))
        width = np.random.choice(np.arange(start=self.width_range[0], stop=self.width_range[1] + 1))
        height = np.random.choice(np.arange(start=self.height_range[0], stop=self.height_range[1] + 1))
        x = np.random.choice(np.arange(start=0, stop=img.shape[0] - width + 1))
        y = np.random.choice(np.arange(start=0, stop=img.shape[1] - height + 1))
        img[y, x:x + width - 1] = 255
        img[y + height - 1, x:x + width] = 255
        img[y:y + height - 1, x] = 255
        img[y:y + height, x + width - 1] = 255
        return self.add_noise(img)

    def draw_cross(self):
        img = np.zeros((self.img_dims, self.img_dims))
        width = np.random.choice(np.arange(start=self.width_range[0], stop=self.width_range[1] + 1))
        height = np.random.choice(np.arange(start=self.height_range[0], stop=self.height_range[1] + 1))
        x = np.random.choice(np.arange(start=0, stop=img.shape[0] - width + 1))
        y = np.random.choice(np.arange(start=0, stop=img.shape[1] - height + 1))
        img[y + round(((height - 1) / 2)), x:x + width] = 255
        img[y:y + height, x + round(((width - 1) / 2))] = 255
        return self.add_noise(img)

    def draw_circle(self):
        pass

    def draw_vertical_bars(self):
        pass

    def draw_horizontal_bars(self):
        pass

    def draw_triangle(self):
        pass

    def add_noise(self, img):
        flat_img = img.reshape(-1)
        idx = np.random.choice(np.arange(start=0, stop=len(flat_img)), round(self.img_noise * len(flat_img)))
        flat_img[idx] = 255 - flat_img[idx]
        return flat_img.reshape(img.shape)

    def generate_dataset(self):
        pass
