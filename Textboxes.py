import textboxes_config as config
import textboxes_prepreocess as preprocess
import textboxes_net as net
import textboxes_bbox as bbox
import image_util
import tensorflow as tf


class Textboxes(object):
    def __init__(self, config_file):
        self._cfg = config.get_config(config_file)

    def _graph(self, input, isTraining):
        return net.make_graph(self._cfg, input, isTraining)

    def train(self):
        None

    def eval(self):
        None

    def inference(self, image):
        image = preprocess.preprocess(self._cfg, image);
        image = tf.expand_dims(image, 0)
        outs_conf, outs_loc = self._graph(image, False)
        return bbox.select(self._cfg, outs_conf, outs_loc)


if __name__ == "__main__":
    import tf_utils
    import image_util

    tb = Textboxes('poc/config.yml')

    image, height, width = image_util.load_image('./poc/ballet_123_4.jpg')
    scores, bboxes = tb.inference(image)
    scores, bboxes = tf_utils.simple_eval(scores, bboxes)

    image_util.DrawImage().image(image, height, width).bbox(bboxes[0][:5],alpha=0.3).draw()
