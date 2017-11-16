import tensorflow as tf
import numpy as np

slim = tf.contrib.slim
Dataset = slim.dataset.Dataset
DatasetDataProvider = slim.dataset_data_provider.DatasetDataProvider

MEAN = [123. / 255., 117. / 256., 104. / 255.]


class Provider(object):
    def __init__(self, config):
        self.config = config
        c = self.config

        descriptions = {
            'image': 'slim.tfexample_decoder.Image',
            'height': 'height',
            'width': 'width',
            'object/bbox': 'box',
            'object/class/label': 'label'
        }

        featureKeys = {
            'image/height': tf.FixedLenFeature([1], tf.int64),
            'image/width': tf.FixedLenFeature([1], tf.int64),
            'image/format': tf.FixedLenFeature([], tf.string, default_value='jpeg'),
            'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
            'image/filename': tf.FixedLenFeature([], dtype=tf.string),
            # 'image/name': tf.FixedLenFeature([], dtype=tf.string),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/class/label': tf.VarLenFeature(dtype=tf.int64)
        }

        handlers = {
            'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
            'height': slim.tfexample_decoder.Tensor('image/height'),
            'width': slim.tfexample_decoder.Tensor('image/width'),
            # image가 transpose되어들어가 있다.
            'object/bbox': slim.tfexample_decoder.BoundingBox(['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
            'object/label': slim.tfexample_decoder.Tensor('image/object/class/label'),
            'filename': slim.tfexample_decoder.Tensor('image/filename')
            # 'filename': slim.tfexample_decoder.Tensor('image/name')
        }

        decoder = slim.tfexample_decoder.TFExampleDecoder(featureKeys, handlers)

        dataset = Dataset(data_sources=c.data.source,
                          reader=tf.TFRecordReader,
                          decoder=decoder,
                          num_samples=c.data.num_samples,
                          num_classes=c.num_classes,
                          items_to_descriptions=descriptions)

        self.provider = DatasetDataProvider(dataset,
                                            num_readers=c.train.num_readers,
                                            common_queue_capacity=c.train.queue_min + 20 * c.train.num_readers,
                                            common_queue_min=c.train.queue_min)

    # 각 layer, 각 점마다 predefined anchor box가 총 12개 ( 6개 ratio * (overlap + center) )
    def def_bboxes(self):
        for i, s in enumerate(self.config.layers):
            y, x = np.mgrid[0:s, 0:s]
            y = (y.astype(np.float32) + 0.5) / s
            x = (x.astype(np.float32) + 0.5) / s
            y_overlap = y + 0.5 / s
            y_out = np.stack((y, y_overlap), -1)
            x_out = np.stack((x, x), -1)
            y_out = np.expand_dims(y_out, axis=-1)
            x_out = np.expand_dims(x_out, axis=-1)
            print(x_out.shape)

    def get(self):
        return self.provider.get(['image', 'object/label', 'object/bbox'])


def main():
    from config import get_config
    import image

    config = get_config("poc/config.yml")
    provider = Provider(config)

    image = provider.get_batch()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with slim.queues.QueueRunners(sess):
            image = sess.run(image)
            image.result_draw((image[0] + MEAN) * 255., None, None)
            image.result_draw((image[1] + MEAN) * 255., None, None)


def foo():
    from config import get_config
    import image

    config = get_config("poc/config.yml")
    provider = Provider(config)
    provider.def_bboxes()


if __name__ == "__main__":
    foo()
