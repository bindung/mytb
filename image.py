import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def draw_bbox(bbox, height, width, **kwarg):
    y1 = bbox[0] * height
    x1 = bbox[1] * width
    y2 = bbox[2] * height
    x2 = bbox[3] * width
    return patches.Rectangle((x1, y1), x2 - x1, y2 - y1, **kwarg)

# image : RGB
def result_draw(image, gt_bboxes, detected_bboxes):
    shape = image.shape
    height = shape[0]
    width = shape[1]
    fig, ax = plt.subplots(1)
    ax.imshow(image.astype(np.uint8))
    if gt_bboxes is not None :
        for bbox in gt_bboxes:
            p = draw_bbox(bbox, height, width, alpha=0.3, color='r')
            ax.add_patch(p)
    plt.show()
