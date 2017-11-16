import tensorflow as tf
import numpy as np
import math
from collections import namedtuple

TextboxParams = namedtuple('TextboxParameters',
                           ['img_shape',
                            'num_classes',
                            'feat_layers',
                            'feat_shapes',
                            'scale_range',
                            'anchor_ratios',
                            'normalizations',
                            'prior_scaling',
                            'anchor_sizes',
                            'anchor_steps',
                            'scales',
                            'match_threshold'])

def textbox_anchor_one_layer(img_shape,
                             feat_size,
                             ratios,
                             dtype=np.float32):
    # Follow the papers scheme
    # 12 ahchor boxes with out sk' = sqrt(sk * sk+1)
    y, x = np.mgrid[0:feat_size[0], 0:feat_size[1]]

    # find center of box
    y = (y.astype(dtype) + 0.5) / feat_size[0]
    x = (x.astype(dtype) + 0.5) / feat_size[1]
    y_overlap = y + 0.5 / feat_size[0] # vertical overlap

    x_out = np.stack((x, x), -1)
    y_out = np.stack((y, y_overlap), -1)

    x_out = np.expand_dims(x_out, axis=-1)
    y_out = np.expand_dims(y_out, axis=-1)

    numofanchor = len(ratios)
    h = np.zeros((numofanchor,), dtype=dtype)
    w = np.zeros((numofanchor,), dtype=dtype)

    for i, r in enumerate(ratios):
        # 가로 세로 비가 r이 되려면 세로는 sqrt(r)로 줄여주고 가로는 sqrt(r)로 늘려주면 된다.
        h[i] = feat_size[0] / img_shape[0] / math.sqrt(r)
        w[i] = feat_size[1] / img_shape[1] * math.sqrt(r)

    return y_out, x_out, h, w

def textbox_achor_all_layers(img_shape,
                             layers_shape,
                             anchor_ratios) :
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = textbox_anchor_one_layer(img_shape,
                                                 s, #feat_size (38,38) or ...
                                                 anchor_ratios)
        layers_anchors.append(anchor_bboxes)
        break
    return layers_anchors


class TextboxNet(object):
    default_params = TextboxParams(
        img_shape=(300, 300),
        num_classes=2,
        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
        anchor_ratios=[1, 2, 3, 5, 7, 10] # 논문에서 제시한 ratio
    )

    def __init__(self, params=None):
        if isinstance(params, TextboxParams):
            self.params = params
        else:
            self.params = self.default_params

    def anchors(self):
        return textbox_achor_all_layers(self.params.img_shape,
                                        self.params.feat_shapes,
                                        self.params.anchor_ratios,
                                        self.params.scales,
                                        self.params.anchor_sizes)


tb = TextboxNet()
tb.anchors()
