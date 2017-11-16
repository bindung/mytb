import tensorflow as tf

slim = tf.contrib.slim
Dataset = slim.dataset.Dataset
DatasetDataProvider = slim.dataset_data_provider.DatasetDataProvider

filename = "sample.tfrecord"
num_samples = 1000


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write():
    with tf.python_io.TFRecordWriter(filename) as writer:
        for i in range(num_samples):
            example = tf.train.Example(features=tf.train.Features(feature={
                'i': int64_feature(i),
                's': int64_feature(i * i)
            }))
            writer.write(example.SerializeToString())


def read():
    desc = {
        'i': 'i',
        's': 's'
    }

    featureKeys = {
        'i': tf.FixedLenFeature([1], tf.int64),
        's': tf.FixedLenFeature([1], tf.int64)
    }

    handlers = {
        'i': slim.tfexample_decoder.Tensor('i'),
        's': slim.tfexample_decoder.Tensor('s')
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(featureKeys, handlers)

    dataset = Dataset(data_sources=filename,
                      reader=tf.TFRecordReader,
                      decoder=decoder,
                      num_samples=num_samples,
                      items_to_descriptions=desc)
    provider = DatasetDataProvider(dataset,
                                   num_readers=1,
                                   common_queue_capacity=512,
                                   common_queue_min=10)
    return provider.get(['i', 's'])


write()
i, s = read()
with tf.Session() as sess:
    a, b = tf.train.shuffle_batch([i, s], batch_size=30, capacity=100, num_threads=1, min_after_dequeue=10)
    sess.run(tf.global_variables_initializer())
    with slim.queues.QueueRunners(sess):
        for i in range(30) :
            c, d = sess.run([a, b])
            print("{} {}".format(c[0], d[0]))
            print("{} {}".format(c[1], d[1]))
            print("{} {}".format(c[2], d[2]))
