import tensorflow as tf
from tensorflow.contrib import slim


def define_graph(config):
    num_layers = 6

    model_cfg = config['model']
    tb_cfg = model_cfg['textbox_layer']

    def calc_scale(k):
        if 'scale' not in tb_cfg[k]:
            min = tb_cfg['min_scale']
            max = tb_cfg['max_scale']

            return min + (max - min) / (num_layers - 1) * k

        return tb_cfg[k]['scale']

    tb_cfg['num_layers'] = num_layers

    tb_cfg[0]['name'] = 'conv4'
    tb_cfg[0]['scale'] = calc_scale(0)
    tb_cfg[0]['shape'] = (38, 38)

    tb_cfg[1]['name'] = 'conv7'
    tb_cfg[1]['scale'] = calc_scale(1)
    tb_cfg[1]['shape'] = (19, 19)

    tb_cfg[2]['name'] = 'conv8'
    tb_cfg[2]['scale'] = calc_scale(2)
    tb_cfg[2]['shape'] = (10, 10)

    tb_cfg[3]['name'] = 'conv9'
    tb_cfg[3]['scale'] = calc_scale(3)
    tb_cfg[3]['shape'] = (5, 5)

    tb_cfg[4]['name'] = 'conv10'
    tb_cfg[4]['scale'] = calc_scale(4)
    tb_cfg[4]['shape'] = (3, 3)

    tb_cfg[5]['name'] = 'global'
    tb_cfg[5]['scale'] = calc_scale(5)
    tb_cfg[5]['shape'] = (1, 1)


def conv_layer(config, b_images):
    model_cfg = config['model']
    tb_cfg = model_cfg['textbox_layer']
    endpoints = []

    # conv1/2 = 300, 300, 64
    conv_net = slim.repeat(b_images, 2, slim.conv2d, 64, [3, 3], scope='conv1')

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
    endpoints.append(conv_net)

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
    endpoints.append(conv_net)

    with tf.variable_scope(tb_cfg[2]['name']):
        conv_net = slim.conv2d(conv_net, 256, [1, 1], scope='conv1x1')
        conv_net = slim.conv2d(conv_net, 512, [3, 3], scope='conv3X3', stride=2)
        endpoints.append(conv_net)

    with tf.variable_scope(tb_cfg[3]['name']):
        conv_net = slim.conv2d(conv_net, 128, [1, 1], scope='conv1x1')
        conv_net = slim.conv2d(conv_net, 256, [3, 3], scope='conv3X3', stride=2)
        endpoints.append(conv_net)

    with tf.variable_scope(tb_cfg[4]['name']):
        conv_net = slim.conv2d(conv_net, 128, [1, 1], scope='conv1x1')
        conv_net = slim.conv2d(conv_net, 256, [3, 3], scope='conv3X3', stride=2)
        endpoints.append(conv_net)

    with tf.variable_scope(tb_cfg[5]['name']):
        conv_net = slim.conv2d(conv_net, 128, [1, 1], scope='conv1x1')
        conv_net = slim.conv2d(conv_net, 256, [3, 3], scope='conv3X3', padding='VALID')
        endpoints.append(conv_net)

    tb_cfg['num_layers'] = len(endpoints)
    return endpoints


def textbox_layer(config, endpoints):
    model_cfg = config['model']
    tb_cfg = model_cfg['textbox_layer']

    b_out_conf = []
    b_out_bbreg = []

    for i in range(tb_cfg['num_layers']):
        layer_cfg = tb_cfg[i]

        num_vertical_offset = len(layer_cfg['vertical_offsets'])
        num_ratios = len(layer_cfg['ratios'])
        num_conf = num_vertical_offset * num_ratios * 2
        num_loc = num_vertical_offset * num_ratios * 4

        conv_out = endpoints[i]
        # TODO: l2 norm
        with tf.variable_scope(layer_cfg['name']):
            # TODO: 한꺼번에 계산하고 split하는게 더 빠를라나?
            b_out_conf.append(slim.conv2d(conv_out,
                                          num_conf,
                                          layer_cfg['kernel_size'],
                                          padding=layer_cfg['padding'],
                                          scope='conf_out'))

            b_out_bbreg.append(slim.conv2d(conv_out,
                                           num_loc,
                                           layer_cfg['kernel_size'],
                                           padding=layer_cfg['padding'],
                                           scope='loc_out'))

    return b_out_conf, b_out_bbreg


def graph(config, b_images, isTraining):
    endpoints = conv_layer(config, b_images)

    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.9997,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        'is_training': isTraining,
        'scale': False,
        'fused': True,
    }

    # with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):

    # return b_out_conf, b_out_bbreg
    return textbox_layer(config, endpoints)


if __name__ == '__main__':
    from textboxes.config import get_config
    from textboxes import Textboxes, prepreocess, data

    config = get_config("poc/config.yml")
    provider = data.Provider(config)
    image, height, width, labels, gboxes = provider.get()
    orig_image = image
    image, labels, gboxes = prepreocess.preprocess_for_train(config, image, height, width, labels, gboxes)

    image = tf.reshape(image, [1, 300, 300, 3])
    tb = Textboxes(config)
    out_conf, out_loc = tb.graph(image, False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with slim.queues.QueueRunners(sess):
            out_conf, out_loc = sess.run([out_conf, out_loc])
