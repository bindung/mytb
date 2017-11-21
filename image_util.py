import io
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.misc import imread

from PIL import Image


def draw_bbox(bbox, height, width, **kwarg):
    y1 = bbox[0] * height
    x1 = bbox[1] * width
    y2 = bbox[2] * height
    x2 = bbox[3] * width
    return patches.Rectangle((x1, y1), x2 - x1, y2 - y1, **kwarg)


class DrawImage(object):
    def __init__(self):
        self.fig, self.ax = plt.subplots(1)
        self.height = None
        self.width = None

    def image(self, image, height, width):
        self.ax.imshow(image)
        self.height = height
        self.width = width
        return self

    def bbox(self, bboxes, alpha=0.2, color='g'):
        for bbox in bboxes:
            p = draw_bbox(bbox, self.height, self.width, alpha=alpha, color=color)
            self.ax.add_patch(p)
        return self

    def draw(self):
        plt.show()


def load_image(image_file):
    image = imread(image_file)
    height = image.shape[0]
    width = image.shape[1]
    return image, height, width


if __name__ == "__main__":
    image, height, width = load_image("./poc/ballet_123_4.jpg")
    DrawImage().image(image, height, width).draw()
