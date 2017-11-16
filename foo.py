import numpy as np
import tensorflow as tf
import tf_utils


class Selected(object):
    def __init__(self):
        self._a = tf.Variable([1.])

    def update(self, m):
        self._a = self._a * m
        return self


a = tf.Variable(2.)
b = tf.Variable(3.)
sel = {'a': a, 'b': b}


def update(s, m):
    return {'a': s['a'] * m, 'b': s['b'] * m}


cond = lambda i, a, b, m: tf.less(i, 3)
body = lambda i, a, b, m: (i + 1, a * m, b * m, m)

cond2 = lambda i, s, m: tf.less(i, 3)
body2 = lambda i, s, m: (i + 1, update(s, m), m)

# i, a, b, m = tf.while_loop(cond, body, [0, a, b, tf.constant(2.)])
i, s, m = tf.while_loop(cond2, body2, [0, sel, tf.constant(5.)])

t = tf.constant([[True, True],[True, True]])
u = tf.reduce_all(t)

print(tf_utils.simple_eval(u))

