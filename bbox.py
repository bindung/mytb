import tensorflow as tf
import math
import numpy as np
import tf_utils


def anchors(model_conf, layer_idx):
    layer_conf = model_conf.textbox_layers[layer_idx]

    """
    index trick.
    conv_size = [width, height]
    y, x = [height, width] 2d array y = [0 1 2 ..] ^ T 가 width만큼 x = [ 0 1 2 ... ] 가 column만큼
    
    >>> y, x = np.mgrid[0:3, 0:2]
    y = 
    [[0 0]
     [1 1]
     [2 2]]
     
    x =
    [[0 1]
     [0 1]
     [0 1]]
    """
    y, x = np.mgrid[0:layer_conf.conv_size[1], 0:layer_conf.conv_size[0]]

    # [0,1) scale. 중심점
    y = (y.astype(np.float32) + 0.5) / layer_conf.conv_size[1]
    x = (x.astype(np.float32) + 0.5) / layer_conf.conv_size[0]

    n = len(model_conf.anchors)

    # anchor 별로 width, height 계산
    h = np.zeros((n,), np.float32)
    w = np.zeros((n,), np.float32)
    for anchor_idx in range(n):
        sqr = math.sqrt(model_conf.anchors[anchor_idx].ratio)
        # layer마다 predefine되어있는 anchor의 크기가 정해져 있다
        h[anchor_idx] = layer_conf.anchor_scale / model_conf.image_size[1] / sqr
        w[anchor_idx] = layer_conf.anchor_scale / model_conf.image_size[0] * sqr
    return y, x, h, w


def find_positive_anchor_box_one_layer(layer_conf, layer_anchors, num_bboxes, gt_bboxes):
    y, x, h, w = layer_anchors

    """"
    numpy broadcasting rule.
    General Broadcasting Rule : they are equal or one of them is 1. align last index
    
    이경우 y, x 가 anchor 갯수만큼 늘어나야한다. rule에 의해 마지막에 1개짜리 index가 늘어나면 된다.
    Y|X = ? X ? X ... X 1
    H|W =               N
    res = ? X ? X ... X N
    """
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    ymin = np.maximum(y - (h / 2.), 0.)
    ymax = np.minimum(y + (h / 2.), 1.)
    xmin = np.maximum(x - (w / 2.), 0.)
    xmax = np.minimum(x + (w / 2.), 1.)

    vol_anchor = (xmax - xmin) * (ymax - ymin)

    feature = {
        'iou': tf.zeros(ymin.shape, tf.float32),
        'ymin': tf.zeros(ymin.shape, tf.float32),
        'xmin': tf.zeros(ymin.shape, tf.float32),
        'ymax': tf.zeros(ymin.shape, tf.float32),
        'xmax': tf.zeros(ymin.shape, tf.float32)
    }

    # while_loop의 body부분. bbox 1 개당 한번 호출된다.
    def body(idx, feat):
        """
        bbox 1개에 대해서 모든 predefined anchor box에 대해서 iou계산.
        threshold값보다 큰 anchor box를 기록해둠.
        단 한 predefined anchor box에 여러개의 bbox가 threshold값을 넘겼다면 그중 가장 큰 bbox만 기억한다.
        기억하는 변수가 feature. 가장 큰 iou값(비교용)과 그 때의 bbox의 값.
        """

        bbox = gt_bboxes[idx]

        # 1. iou 계산
        with tf.name_scope("calculate_iou"):
            iymin = tf.maximum(ymin, bbox[0])
            ixmin = tf.maximum(xmin, bbox[1])
            iymax = tf.minimum(ymax, bbox[2])
            ixmax = tf.minimum(xmax, bbox[3])

            vol_box = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            # 겹치는 영역이 없는 경우 max - min은 음수가 되므로 이를 고려
            vol_inter = tf.maximum(iymax - iymin, 0.) * tf.maximum(ixmax - ixmin, 0)
            vol_union = vol_anchor + vol_box - vol_inter
            iou = tf.div(vol_inter, vol_union)

        # feature update하는 함수
        def feature_update(feat, mask):
            fmask = tf.cast(mask, tf.float32)
            return {
                # _iou = mask ? iou : _iou
                'iou': tf.where(mask, iou, feat['iou']),

                # _ymin = mask ? bbox[0] : _ymin
                # 단 bbox[0]은 (1,)이고 fmask는 n-dim tensor이므로 형태를 맞추기 위해서.
                'ymin': fmask * bbox[0] + (1. - fmask) * feat['ymin'],
                'xmin': fmask * bbox[1] + (1. - fmask) * feat['xmin'],
                'ymax': fmask * bbox[2] + (1. - fmask) * feat['ymax'],
                'xmax': fmask * bbox[3] + (1. - fmask) * feat['xmax']
            }

        mask = (iou > feat['iou']) & (iou > layer_conf.iou_threshold)

        # 찾은 box update
        feat = feature_update(feat, mask)

        max_iou = tf.reduce_max(iou)

        feat = tf.cond(
            # 새로찾은 bbox도 없고 && 제일 큰 iou가 min_threshold이상이라면 그것을 feature에 추가
            ~tf.reduce_all(mask) & (max_iou > layer_conf.min_threshold),
            lambda: feature_update(feat, tf.equal(iou, max_iou)),
            lambda: feat
        )

        return idx + 1, feat

    # 2 모든 bbox에 대해서 찾자.
    _, feature = tf.while_loop(
        lambda i, _: i < num_bboxes,
        body,
        [0, feature]
    )
    return feature


# (layer, bbox)당 한 개의 max_iou(>threshold)를 가진 bbox를 찾는다.
def find_positive_anchor_box(model_conf, labels, gt_bboxes):
    # TODO : change input data type
    labels = tf.cast(labels, tf.int32)
    num_bboxes = tf.reduce_sum(labels)
    features = []
    for i, layer_conf in enumerate(model_conf.textbox_layers):
        feature = find_positive_anchor_box_one_layer(layer_conf,
                                                     anchors(model_conf, i),
                                                     num_bboxes,
                                                     gt_bboxes)
        features.append(feature)
    return features


def main():
    from config import get_config
    from dataLoader import Provider
    from tensorflow.python import debug as tf_debug

    slim = tf.contrib.slim

    config = get_config("poc/config.yml")
    provider = Provider(config)

    _, labels, gt_bboxes = provider.get()

    features = find_positive_anchor_box(config.model, labels, gt_bboxes)

    from tensorflow.python import debug as tf_debug

    def test():
        f, g = sess.run([features, gt_bboxes])
        print(f)

    with tf.Session() as sess:
        with slim.queues.QueueRunners(sess):
            sess.run(tf.global_variables_initializer())
            test()


if __name__ == "__main__":
    main()
