import tensorflow as tf

import tf_image


def preprocess(config, image):
    model_cfg = config['model']
    image_size = model_cfg['image_size']

    with tf.name_scope('preprocess'):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf_image.resize_image(image, image_size)
        image = tf_image.tf_image_whitened(image)

    return image


def preprocess_for_train(config, image, height, width, labels, bboxes):
    model_cfg = config['model']
    image_size = model_cfg['image_size']

    with tf.name_scope('distort'):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # color
        image = tf_image.distorter(image)
        image = tf.clip_by_value(image, 0., 1.)
        image = tf_image.tf_image_whitened(image)

        # crop
        # image, bboxes = tf_image.resize_image_bboxes_with_crop_or_pad2(image, bboxes, height[0], width[0])
        # TODO: Random_crop시 channel 정보가 사라지는데 이거 복원 시키자.
        # image, labels, bboxes = tf_image.Random_crop(image, labels, bboxes)
        # image = tf.reshape(image, list(image_size) + [3])
        image = tf_image.resize_image(image, image_size)
        # image, bboxes = tf_image.random_flip_left_right(image, bboxes)

    return image, labels, bboxes


def test_for_train():
    from textboxes import data
    from textboxes.config import get_config
    from image_util import DrawImage

    slim = tf.contrib.slim

    config = get_config("poc/config.yml")
    provider = data.Provider(config)
    image, height, width, labels, gboxes = provider.get()

    image, labels, gboxes = preprocess_for_train(config, image, height, width, labels, gboxes)
    image = tf_image.tf_image_unwhitened(image)
    with tf.Session() as sess:
        with slim.queues.QueueRunners(sess):
            image, height, width, gboxes = sess.run([image, height, width, gboxes])

    DrawImage().image(image, 300, 300).bbox(gboxes, 0.3, 'r').draw()


def test_for_eval():
    from textboxes.config import get_config
    import image_util

    config = get_config("poc/config.yml")
    image, height, width = image_util.load_image("./poc/ballet_123_4.jpg")
    image = preprocess(config, image)
    image = tf_image.tf_image_unwhitened(image)
    with tf.Session() as sess:
        image = sess.run(image)

    image_util.DrawImage().image(image, 300, 300).draw()


if __name__ == '__main__':
    test_for_train()
    test_for_eval()
