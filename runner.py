import tensorflow as tf
import textboxes.config as config
import textboxes.net as net
import textboxes.prepreocess as pproc
import textboxes.bbox as bbox
import textboxes.loss as loss
from textboxes.data import Provider


class Runner(object):
    def __init__(self, config_file):
        self._cfg = config.get_config(config_file)
        net.define_graph(self._cfg)

    def _graph(self, input, isTraining):
        return net.graph(self._cfg, input, isTraining)

    def train(self):
        provider = Provider(self._cfg)

        train_conf = self._cfg['train']
        tb_conf = self._cfg['model']['textbox_layer']
        batch_size = train_conf['batch_size']

        image, height, width, labels, gt_bboxes = provider.get()
        image, labels, gt_bboxes = pproc.preprocess_for_train(self._cfg, image, height, width, labels, gt_bboxes)

        # shape = [LAYERS][....]
        iou, gt_bbox_reg = bbox.encode(self._cfg, labels, gt_bboxes, format=bbox.BBOX_REG)

        # iou, gt_loc 는 [layers][...]이고 각 layer들이 shape이 모두 다르다
        # batch를 만들때 image, iou[0], iou[1], ..., gt_loc[0], gt_loc[1], ... 로 풀어서 배치 만든다음에 다시 조립하자.

        batch_shape = [1] + [len(iou)] + [len(gt_bbox_reg)]

        def reshape_to_list(li):
            ret = []
            for i in li:
                if isinstance(i, (list, tuple)):
                    ret = ret + list(i)
                else:
                    ret.append(i)

            return ret

        def reshape_from_list(li, shape):
            ret = []
            idx = 0
            for s in shape:
                if s == 1:
                    ret.append(li[idx])
                else:
                    ret.append(li[idx:idx + s])
                idx += s
            return ret

        b_image, b_iou, b_gt_bbox_reg = reshape_from_list(
            tf.train.batch(
                reshape_to_list([image, iou, gt_bbox_reg]),
                batch_size=batch_size,
                allow_smaller_final_batch=True
            ),
            [1] + [len(iou)] + [len(gt_bbox_reg)]
        )

        b_out_conf, b_out_bbox_reg = net.graph(self._cfg, b_image, isTraining=True)
        total_loss = loss.calculate_loss(self._cfg, b_out_conf, b_out_bbox_reg, b_iou, b_gt_bbox_reg)

        optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)
        train_op = slim.learning.create_train_op(
            total_loss,
            optimizer,
            summarize_gradients=True
        )

        slim.learning.train(
            train_op,
            'train/exp1',
            save_summaries_secs=20
        )

    def eval(self):
        None

    def inference(self, image):
        image = pproc.preprocess(self._cfg, image)
        image = tf.expand_dims(image, 0)
        outs_conf, outs_loc = self._graph(image, False)
        return bbox.select(self._cfg, outs_conf, outs_loc)


def test_inference():
    import image_util
    import tf_utils

    runner = Runner("poc/config.yml")
    image, height, width = image_util.load_image('poc/ballet_123_4.jpg')
    scores, bboxes = runner.inference(image)
    scores, bboxes = tf_utils.simple_eval(scores, bboxes)
    image_util.DrawImage().image(image, height, width).bbox(bboxes[0][:5], alpha=0.3, color='g').draw()


if __name__ == "__main__":
    import tf_utils

    slim = tf.contrib.slim

    runner = Runner("poc/config.yml")
    runner.train()
