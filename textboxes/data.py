import tensorflow as tf

slim = tf.contrib.slim
Dataset = slim.dataset.Dataset
DatasetDataProvider = slim.dataset_data_provider.DatasetDataProvider


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
            'object/bbox': slim.tfexample_decoder.BoundingBox(['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
            'object/label': slim.tfexample_decoder.Tensor('image/object/class/label')
        }

        decoder = slim.tfexample_decoder.TFExampleDecoder(featureKeys, handlers)

        data_cfg = self._cfg['data']
        dataset = Dataset(data_sources=data_cfg['source'],
                          reader=tf.TFRecordReader,
                          decoder=decoder,
                          num_samples=data_cfg['num_samples'],
                          num_classes=2,
                          items_to_descriptions=descriptions)

        self.provider = DatasetDataProvider(dataset,
                                            common_queue_capacity=32,
                                            common_queue_min=8)

    def get(self):
        return self.provider.get(['image', 'height', 'width', 'object/label', 'object/bbox'])
