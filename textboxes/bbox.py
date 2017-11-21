import tensorflow as tf
import math
import numpy as np
import tensorflow.contrib.slim as slim

"""
    gboxes: ground truth bounding box의 목록. 갯수가 고정되어있지 않다. 
        shape = [?][4=(ymin,xmin,ymax,ymax)] (iou 계산하기 편하라고 cy, cx, h, w notation이 아님)
        [0,1] scale
    
    dbox: default(=predefined) bounding box
"""

"""
def loc_to_bbox(config, loc):
    s = [slice(None, None, None) for i in range(len(loc.get_shape()) - 1)]

    dy = loc.__getitem__(s + [0])
    dx = loc.__getitem__(s + [1])
    dh = loc.__getitem__(s + [2])
    dw = loc.__getitem__(s + [3])

    y1 = cy - h / 2.
    y2 = cy + h / 2.
    x1 = cx - w / 2.
    x2 = cx + w / 2.
    return tf.stack([y1, x1, y2, x2], -1)
"""

BBOX = 0
CENTERED = 1
BBOX_REG = 2


def anchors(layer_cfg):
    shape = layer_cfg['shape']

    y, x = np.mgrid[0:shape[0], 0:shape[1]]

    y = y.astype(np.float32) + 0.5
    x = x.astype(np.float32) + 0.5

    # TODO: vertical offset
    dbox_cy = np.stack((y, y + 0.5), -1) / shape[0]
    dbox_cx = np.stack((x, x), -1) / shape[1]

    # width, height 계산
    sq_ratios = [math.sqrt(ratio) for ratio in layer_cfg['ratios']]
    dbox_h = np.array([layer_cfg['scale'] / sqr for sqr in sq_ratios], dtype=np.float32)
    dbox_w = np.array([layer_cfg['scale'] * sqr for sqr in sq_ratios], dtype=np.float32)

    return dbox_cy, dbox_cx, dbox_h, dbox_w


def encode_one_layer(layer_cfg, num_gt_gboxes, gt_gboxes, format=CENTERED):
    """
    :param layer_cfg: config
    :param num_gt_gboxes:
    :param gt_gboxes:
    :return: iou, location(shape=[?,...][4])
    """

    with tf.name_scope("def_anchor_box"):
        dbox_cy, dbox_cx, dbox_h, dbox_w = anchors(layer_cfg)

        dbox_cy = np.expand_dims(dbox_cy, axis=-1)
        dbox_cx = np.expand_dims(dbox_cx, axis=-1)

        dbox_y1 = np.maximum(dbox_cy - (dbox_h / 2.), 0.)
        dbox_x1 = np.maximum(dbox_cx - (dbox_w / 2.), 0.)
        dbox_y2 = np.minimum(dbox_cy + (dbox_h / 2.), 1.)
        dbox_x2 = np.minimum(dbox_cx + (dbox_w / 2.), 1.)

        vol_anchor = (dbox_x2 - dbox_x1) * (dbox_y2 - dbox_y1)

    max_iou = tf.zeros(dbox_y1.shape, tf.float32)
    gt_bbox_y1 = tf.zeros(dbox_y1.shape, tf.float32)
    gt_bbox_x1 = tf.zeros(dbox_y1.shape, tf.float32)
    gt_bbox_y2 = tf.zeros(dbox_y1.shape, tf.float32)
    gt_bbox_x2 = tf.zeros(dbox_y1.shape, tf.float32)

    def body(idx, max_iou, gbox_y1, gbox_x1, gbox_y2, gbox_x2):
        """
        iou계산해서 저장해둔 값보다 크다면 업데이트한다.
        """

        gbox = gt_gboxes[idx]

        # 1. iou 계산
        with tf.name_scope("calc_iou"):
            i_y1 = tf.maximum(dbox_y1, gbox[0])
            i_x1 = tf.maximum(dbox_x1, gbox[1])
            i_y2 = tf.minimum(dbox_y2, gbox[2])
            i_x2 = tf.minimum(dbox_x2, gbox[3])

            vol_box = (gbox[2] - gbox[0]) * (gbox[3] - gbox[1])
            # 겹치는 영역이 없는 경우 max - min은 음수가 되므로 이를 고려
            vol_inter = tf.maximum(i_y2 - i_y1, 0.) * tf.maximum(i_x2 - i_x1, 0)
            vol_union = vol_anchor + vol_box - vol_inter
            iou = tf.div(vol_inter, vol_union)

        def _update(mask):
            fmask = tf.cast(mask, tf.float32)

            _i = tf.where(mask, iou, max_iou)
            _y1 = fmask * gbox[0] + (1. - fmask) * gbox_y1
            _x1 = fmask * gbox[1] + (1. - fmask) * gbox_x1
            _y2 = fmask * gbox[2] + (1. - fmask) * gbox_y2
            _x2 = fmask * gbox[3] + (1. - fmask) * gbox_x2
            return _i, _y1, _x1, _y2, _x2

        # >, & 모두 elementwise 연산이고 tf에서 override되어있다.
        # tf.greater, tf.logical_and
        mask = (iou > max_iou) & (iou > layer_cfg['iou_threshold'])

        # 찾은 anchor box 추가
        max_iou, gbox_y1, gbox_x1, gbox_y2, gbox_x2 = _update(mask)

        # threshold를 넘긴 값이 anchorbox가 없을 때는 가장 큰 것을 found에 추가.
        # 이게 없으면 num_pos가 0이되어 계산에 문제가 생길 수 있다. => 사실상 벌어지지 않는 일.
        # TODO: 근거를 찾자.
        """
        v = tf.reduce_max(iou)
        max_iou, gt_bbox_y1, gt_bbox_x1, gt_bbox_y2, gt_bbox_x2 = \
            tf.cond(
                ~tf.reduce_all(mask),
                lambda: _update(tf.equal(iou, v)),
                lambda: (max_iou, gt_bbox_y1, gt_bbox_x1, gt_bbox_y2, gt_bbox_x2)
            )
        """

        return idx + 1, max_iou, gbox_y1, gbox_x1, gbox_y2, gbox_x2

    # 2 모든 gbox에 대해서 찾자.
    _, max_iou, gt_bbox_y1, gt_bbox_x1, gt_bbox_y2, gt_bbox_x2 = \
        tf.while_loop(
            lambda i, *_: i < num_gt_gboxes,
            body,
            [0, max_iou, gt_bbox_y1, gt_bbox_x1, gt_bbox_y2, gt_bbox_x2]
        )

    if format == BBOX:
        return max_iou, tf.stack((gt_bbox_y1, gt_bbox_x1, gt_bbox_y2, gt_bbox_x2), -1)

    gt_bbox_cy = (gt_bbox_y2 + gt_bbox_y1) / 2.
    gt_bbox_cx = (gt_bbox_x2 + gt_bbox_x1) / 2.
    gt_bbox_h = (gt_bbox_y2 - gt_bbox_y1)
    gt_bbox_w = (gt_bbox_x2 - gt_bbox_x1)

    if format == CENTERED:
        return max_iou, tf.stack((gt_bbox_cy, gt_bbox_cx, gt_bbox_h, gt_bbox_w), -1)

    gt_bbox_reg_dy = (gt_bbox_cy - dbox_cy) / dbox_h
    gt_bbox_reg_dx = (gt_bbox_cx - dbox_cx) / dbox_w
    gt_bbox_reg_dh = tf.log(gt_bbox_h / dbox_h)
    gt_bbox_reg_dw = tf.log(gt_bbox_w / dbox_w)
    return max_iou, tf.stack((gt_bbox_reg_dy, gt_bbox_reg_dx, gt_bbox_reg_dh, gt_bbox_reg_dw), -1)


def encode(config, labels, gt_bboxes, format=CENTERED):
    model_cfg = config['model']

    labels = tf.cast(labels, tf.int32)
    num_gboxes = tf.reduce_sum(labels)

    iou = []
    bbox = []
    tb_cfg = model_cfg['textbox_layer']
    for i in range(tb_cfg['num_layers']):
        layer_cfg = tb_cfg[i]
        with tf.name_scope("encode_{}".format(i)):
            l_iou, l_bbox = encode_one_layer(layer_cfg,
                                             num_gboxes,
                                             gt_bboxes,
                                             format=format)
            iou.append(l_iou)
            bbox.append(l_bbox)
    return iou, bbox


def decode_one_layer(layer_cfg, out_loc):
    dbox_cy, dbox_cx, dbox_h, dbox_w = anchors(layer_cfg)

    num_vertical_offsets = len(layer_cfg['vertical_offsets'])
    num_ratios = len(layer_cfg['ratios'])

    # shape = [b][y][x][2 * 6 * 4] => [b][y][x][2][6][4]
    shape = out_loc.get_shape().as_list()
    out_loc = tf.reshape(out_loc, (shape[0], shape[1], shape[2], num_vertical_offsets, num_ratios, 4))

    # d?.shape = [...][6]
    dy = out_loc[:, :, :, :, :, 0]
    dx = out_loc[:, :, :, :, :, 1]
    dh = out_loc[:, :, :, :, :, 2]
    dw = out_loc[:, :, :, :, :, 3]

    # c?.shape = [...][2][6]
    cy = dy * dbox_h + tf.expand_dims(dbox_cy, axis=-1)
    cx = dx * dbox_w + tf.expand_dims(dbox_cx, axis=-1)
    h = tf.exp(dh) * dbox_h
    w = tf.exp(dw) * dbox_w

    ymin = tf.clip_by_value(cy - h / 2., 0., 1.)
    xmin = tf.clip_by_value(cx - w / 2., 0., 1.)
    ymax = tf.clip_by_value(cy + h / 2., 0., 1.)
    xmax = tf.clip_by_value(cx + w / 2., 0., 1.)
    # return.shape = [....][2][6][4]
    return tf.stack([ymin, xmin, ymax, xmax], axis=-1)


def select(cfg, outs_conf, outs_loc):
    model_cfg = cfg['model']
    eval_cfg = cfg['eval']

    tb_cfg = model_cfg['textbox_layer']
    num_layers = tb_cfg['num_layers']
    num_batch = outs_conf[0].get_shape().as_list()[0]

    all_scores = []
    all_bboxes = []
    for i in range(num_layers):
        # shape = [b][y][x][2 * 6 * 2] => [b][y * x * 2 * 6][2]
        conf = tf.reshape(outs_conf[i], (num_batch, -1, 2))

        scores = slim.softmax(conf)[:, :, 1]

        bboxes = decode_one_layer(tb_cfg[i], outs_loc[i])
        bboxes = tf.reshape(bboxes, (num_batch, -1, 4))

        fmask = tf.cast(tf.greater_equal(scores, eval_cfg['min_score']), scores.dtype)
        scores = scores * fmask
        all_scores.append(scores)
        all_bboxes.append(bboxes)

    all_scores = tf.concat(all_scores, axis=1)
    all_bboxes = tf.concat(all_bboxes, axis=1)

    top_k = eval_cfg['top_k']
    nms_threshold = eval_cfg['nms_threshold']
    keep_top_k = eval_cfg['keep_top_k']

    top_scores, idxes = tf.nn.top_k(all_scores, k=top_k, sorted=True)

    top_bboxes = tf.map_fn(lambda x: [tf.gather(x[0], x[1])],
                           [all_bboxes, idxes],
                           dtype=[bboxes.dtype])[0]

    def nms(scores, bboxes):
        idxes = tf.image.non_max_suppression(bboxes, scores, keep_top_k, nms_threshold)
        scores = tf.gather(scores, idxes)
        bboxes = tf.gather(bboxes, idxes)

        return scores, bboxes

    return tf.map_fn(lambda x: nms(x[0], x[1]),
                     [top_scores, top_bboxes],
                     dtype=(scores.dtype, bboxes.dtype))
