import tensorflow as tf
from tensorflow.contrib import slim

batch_norm_params = {
    # Decay for the moving averages.
    'decay': 0.9997,
    # epsilon to prevent 0s in variance.
    'epsilon': 0.001,
    'is_training': True,
    'scale': False,
    'fused': True,
}


def make_net(config, inputs):
    endpoints = {}
    shapes = []
    with tf.variable_scope('txtbox_300', [inputs]):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
            # conv1/2 = 300, 300, 64
            conv_net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')

            # pool1 = 150, 150, 64
            conv_net = slim.max_pool2d(conv_net, [2, 2], scope='pool1')

            # conv2/2 = 150, 150, 128
            conv_net = slim.repeat(conv_net, 2, slim.conv2d, 128, [3, 3], scope='conv2')

            # pool2 = 75, 75, 128
            conv_net = slim.max_pool2d(conv_net, [2, 2], scope='pool2')

            # conv3/3 = 75, 75, 256
            conv_net = slim.repeat(conv_net, 3, slim.conv2d, 256, [3, 3], scope='conv3')

            # pool3 = 38, 38, 256
            conv_net = slim.max_pool2d(conv_net, [2, 2], scope='pool3', padding='SAME')

            # conv4 = 38, 38, 512
            conv_net = slim.repeat(conv_net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            endpoints['conv4'] = conv_net
            shapes.append([38, 38])

            # pool4 = 19, 19, 512
            conv_net = slim.max_pool2d(conv_net, [2, 2], scope='pool4')

            # conv5 = 19, 19, 512
            conv_net = slim.repeat(conv_net, 3, slim.conv2d, 512, [3, 3], scope='conv5')

            # pool5 = 19, 19, 512
            conv_net = slim.max_pool2d(conv_net, [2, 2], stride=1, scope='pool5', padding='SAME')

            # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L929 
            # slim.conv2d default params
            # rate = 1
            # stride = 1, padding = 'SAME', activation_fn = nn.relu, normalizer_fn=None, normalizer_params=None
            # weights_initializer=initializers.xavier_initializer(), weights_regulairzer = None
            # biases_initializer = init_ops.zeros_initializer(), biases_regularizer = None
            # trinable = True

            # stride = 1, padding='SAME', rate=6, 
            # 참고로 padding이 VALID이면 7,7 이 된다. output의 0번 index가 필요한 input의 마지막 index가 12이므로 19-12 = 7 (stride = 1)
            # conv6 = 19, 19, 1024
            conv_net = slim.conv2d(conv_net, 1024, [3, 3], scope='conv6', rate=6)

            # conv7 = 19, 19, 1024
            conv_net = slim.conv2d(conv_net, 1024, [1, 1], scope='conv7')
            endpoints['conv7'] = conv_net
            shapes.append([19, 19])

            scope = 'conv8'
            with tf.variable_scope(scope):
                conv_net = slim.conv2d(conv_net, 256, [1, 1], scope='conv1x1')
                conv_net = slim.conv2d(conv_net, 512, [3, 3], scope='conv3X3', stride=2)
                endpoints[scope] = conv_net
                shapes.append([10, 10])

            scope = 'conv9'
            with tf.variable_scope(scope):
                conv_net = slim.conv2d(conv_net, 128, [1, 1], scope='conv1x1')
                conv_net = slim.conv2d(conv_net, 256, [3, 3], scope='conv3X3', stride=2)
                endpoints[scope] = conv_net
                shapes.append([5, 5])

            scope = 'conv10'
            with tf.variable_scope(scope):
                conv_net = slim.conv2d(conv_net, 128, [1, 1], scope='conv1x1')
                conv_net = slim.conv2d(conv_net, 256, [3, 3], scope='conv3X3', stride=2)
                endpoints[scope] = conv_net
                shapes.append([3, 3])

            scope = 'global'
            with tf.variable_scope(scope):
                conv_net = slim.conv2d(conv_net, 128, [1, 1], scope='conv1x1')
                conv_net = slim.conv2d(conv_net, 256, [3, 3], scope='conv3X3', padding='VALID')
                endpoints[scope] = conv_net
                shapes.append([1, 1])

            # (2 = original + overlap position) X (anchorbox = 6) X (2 = classifier + 4 = coordinate)
            num_pred = 2 * len(config.anchor_ratios) * (2 + 4)

            outputs = []
            for scope in endpoints:
                inputs = endpoints[scope]
                # TODO: l2 norm
                with tf.variable_scope(scope):
                    if scope == 'global':
                        kernel_size = [1, 1]
                        padding = 'VALID'
                    else:
                        kernel_size = [1, 5]
                        padding = 'SAME'

                    out = slim.conv2d(inputs, num_pred, kernel_size, padding=padding, scope='conv_out')
                    outputs.append(out)
    return outputs


if __name__ == '__main__':
    from config import get_config
    import image
    import dataLoader

    config = get_config("poc/config.yml")
    provider = dataLoader.Provider(config)

    image = provider.get_batch()
    outputs = make_net(config, image)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with slim.queues.QueueRunners(sess):
            outputs = sess.run(outputs)
            print(outputs[0].shape)
