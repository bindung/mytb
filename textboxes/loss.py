import tensorflow as tf
import tf_utils


def calculate_loss(*args):
    with tf.name_scope("calc_loss"):
        return _calc_loss(*args)


def _calc_loss(config, b_out_conf, b_out_bbox_reg, b_iou, b_gt_bbox_reg):
    """
    :param config:
    :param b_out_conf: network에서 계산한 결과값. shape = [layer][batch][...][6][2]
    :param b_out_bbox_reg: network에서 계산한 결과값. shape = [layer][batch][...][6][4]
    :param b_iou: 그 anchor box과 모든 gt_bboxes의 iou중에 가장 큰 놈. 단 threshold보다 큰 값들만 들어있다. shape = [layer][batch][...][6]
    :param b_gt_bbox_reg: 그 때의 gt_bbox의 [cy, cx, h, w] shape = [layer][batch][...][6][4]
    :return:
    """

    train_cfg = config['train']

    # 일단 첫번째 index(=layer)를 펴자. 안그러면 layer 마다 shape이 달라서 계산이 안된다.
    # b_bbox_reg.shape = [?][4]
    # b_gt_loc.shape = [?][4]
    b_conf = tf.concat([tf.reshape(c, [-1, 2]) for c in b_out_conf], 0)
    b_bbox_reg = tf.concat([tf.reshape(l, [-1, 4]) for l in b_out_bbox_reg], 0)
    b_iou = tf.concat([tf.reshape(i, [-1]) for i in b_iou], 0)
    b_gt_bbox_reg = tf.concat([tf.reshape(l, [-1, 4]) for l in b_gt_bbox_reg], 0)

    # 전체 anchor box 갯수
    num_tot = tf.reduce_sum(tf.ones_like(b_iou, dtype=tf.float32))

    with tf.name_scope("loss_pos"):
        # 일단 iou에는 threshold조건을 만족하는 애들만 들어있으므로 값이 양수이면 positive sample
        pmask = b_iou > 0
        fpmask = tf.cast(pmask, tf.float32)

        # positive sample의 갯수
        num_pos = tf.reduce_sum(fpmask)

        conf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=b_conf, labels=tf.cast(pmask, tf.int32))
        conf_pos_loss = conf_loss * fpmask

    with tf.name_scope("loss_neg"):
        # 학습에 참여할 negative sample의 갯수
        num_neg = tf.minimum(train_cfg['negative_ratio'] * num_pos, num_tot - num_pos)
        conf_loss = conf_loss * (1. - fpmask)  # positive loss는 제거하고 난 나머지 negative 후보들

        # top k(=num_pos * negative_ratio) 개만 학습에 참여
        # image별로 찾는게 아니라 이 배치 전체에서 찾는다.
        vals, _ = tf.nn.top_k(tf.reshape(conf_loss, [-1]), k=tf.cast(num_neg, tf.int32))
        nmask = conf_loss >= vals[-1]
        conf_neg_loss = conf_loss * tf.cast(nmask, tf.float32)

    with tf.name_scope("loss_loc"):
        flmask = tf.expand_dims(fpmask, axis=-1)

        # smooth l1 norm
        abs_diff = tf.abs(b_bbox_reg - b_gt_bbox_reg)
        smooth_l1_norm = tf.where(
            tf.less(abs_diff, 1),
            0.5 * tf.square(abs_diff),
            abs_diff - 0.5
        )
        loc_loss = tf.reduce_sum(smooth_l1_norm * flmask)

    total_loss = (conf_pos_loss + conf_neg_loss + loc_loss) / num_pos

    return total_loss
