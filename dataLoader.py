import tensorflow as tf
import numpy as np

slim = tf.contrib.slim
Dataset = slim.dataset.Dataset
DatasetDataProvider = slim.dataset_data_provider.DatasetDataProvider

MEAN = [123. / 255., 117. / 256., 104. / 255.]


class Provider(object):
    def __init__(self, config):
        self._cfg = config

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

        data_cfg = self._cfg['data']
        dataset = Dataset(data_sources=data_cfg['source'],
                          reader=tf.TFRecordReader,
                          decoder=decoder,
                          num_samples=data_cfg['num_samples'],
                          num_classes=2,
                          items_to_descriptions=descriptions)

        train_cfg = self._cfg['train']
        num_readers = train_cfg['num_readers']
        queue_min = train_cfg['queue_min']
        self.provider = DatasetDataProvider(dataset,
                                            num_readers=num_readers,
                                            common_queue_capacity=queue_min + 20 * num_readers,
                                            common_queue_min=queue_min)

    def get(self):
        return self.provider.get(['image', 'height', 'width', 'object/label', 'object/bbox'])
