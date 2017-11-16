import tensorflow as tf

foo = tf.Variable((0., 1., 2., 3.))
bar = tf.ones((4,))


def cond(index, x, y):
    return tf.less(index, 3)


def body(index, x, y):
    return index + 1, x * y, y


j, bar, _ = tf.while_loop(cond, body, [0, bar, foo])

r = tf.cast(tf.equal(tf.range(5), 1), tf.int32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    r = sess.run([r])
    print(r)
