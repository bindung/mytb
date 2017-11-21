import tensorflow as tf
from tensorflow.contrib import slim

NUM_LAYERS = 6


def make_conv_layer(config, inputs):
    model_cfg = config['model']
    tb_cfg = model_cfg['textbox_layer']
    endpoints = []

    def calc_scale(k):
        if 'scale' not in tb_cfg[k]:
            min = tb_cfg['min_scale']
            max = tb_cfg['max_scale']

            return min + (max - min) / (NUM_LAYERS - 1) * k

        return tb_cfg[k]['scale']

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
    tb_cfg[0]['name'] = 'conv4'
    tb_cfg[0]['scale'] = calc_scale(0)
    tb_cfg[0]['shape'] = (38, 38)
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
    tb_cfg[1]['name'] = 'conv7'
    tb_cfg[1]['scale'] = calc_scale(1)
    tb_cfg[1]['shape'] = (19, 19)
    endpoints.append(conv_net)

    scope = 'conv8'
    with tf.variable_scope(scope):
        conv_net = slim.conv2d(conv_net, 256, [1, 1], scope='conv1x1')
        conv_net = slim.conv2d(conv_net, 512, [3, 3], scope='conv3X3', stride=2)
        tb_cfg[2]['name'] = scope
        tb_cfg[2]['scale'] = calc_scale(2)
        tb_cfg[2]['shape'] = (10, 10)
        endpoints.append(conv_net)

    scope = 'conv9'
    with tf.variable_scope(scope):
        conv_net = slim.conv2d(conv_net, 128, [1, 1], scope='conv1x1')
        conv_net = slim.conv2d(conv_net, 256, [3, 3], scope='conv3X3', stride=2)
        tb_cfg[3]['name'] = scope
        tb_cfg[3]['scale'] = calc_scale(3)
        tb_cfg[3]['shape'] = (5, 5)
        endpoints.append(conv_net)

    scope = 'conv10'
    with tf.variable_scope(scope):
        conv_net = slim.conv2d(conv_net, 128, [1, 1], scope='conv1x1')
        conv_net = slim.conv2d(conv_net, 256, [3, 3], scope='conv3X3', stride=2)
        tb_cfg[4]['name'] = scope
        tb_cfg[4]['scale'] = calc_scale(4)
        tb_cfg[4]['shape'] = (3, 3)
        endpoints.append(conv_net)

    scope = 'global'
    with tf.variable_scope(scope):
        conv_net = slim.conv2d(conv_net, 128, [1, 1], scope='conv1x1')
        conv_net = slim.conv2d(conv_net, 256, [3, 3], scope='conv3X3', padding='VALID')
        tb_cfg[5]['name'] = scope
        tb_cfg[5]['scale'] = calc_scale(5)
        tb_cfg[5]['shape'] = (1, 1)
        endpoints.append(conv_net)

    tb_cfg['num_layers'] = len(endpoints)
    return endpoints


def make_textbox_layer(config, endpoints):
    model_cfg = config['model']
    tb_cfg = model_cfg['textbox_layer']

    out_conf = []
    out_loc = []
    for i in range(NUM_LAYERS):
        layer_cfg = tb_cfg[i]

        num_vertical_offset = len(layer_cfg['vertical_offsets'])
        num_ratios = len(layer_cfg['ratios'])
        num_conf = num_vertical_offset * num_ratios * 2
        num_loc = num_vertical_offset * num_ratios * 4

        inputs = endpoints[i]
        # TODO: l2 norm
        with tf.variable_scope(layer_cfg['name']):
            # TODO: 한꺼번에 계산하고 split하는게 더 빠를라나?
            out_conf.append(
                slim.conv2d(inputs,
                            num_conf,
                            layer_cfg['kernel_size'],
                            padding=layer_cfg['padding'],
                            scope='conf_out')
            )

            out_loc.append(
                slim.conv2d(inputs,
                            num_loc,
                            layer_cfg['kernel_size'],
                            padding=layer_cfg['padding'],
                            scope='loc_out')
            )
    return out_conf, out_loc


def make_graph(config, input, isTraining):
    endpoints = make_conv_layer(config, input)

    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.9997,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        'is_training': isTraining,
        'scale': False,
        'fused': True,
    }

    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
        return make_textbox_layer(config, endpoints)


if __name__ == '__main__':
    from textboxes_config import get_config
    import dataLoader, textboxes_prepreocess
    from Textboxes import Textboxes

    config = get_config("poc/config.yml")
    provider = dataLoader.Provider(config)
    image, height, width, labels, gboxes = provider.get()
    orig_image = image
    image, labels, gboxes = textboxes_prepreocess.preprocess_for_train(config, image, height, width, labels, gboxes)

    image = tf.reshape(image, [1, 300, 300, 3])
    tb = Textboxes(config)
    out_conf, out_loc = tb.graph(image, False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with slim.queues.QueueRunners(sess):
            out_conf, out_loc = sess.run([out_conf, out_loc])
