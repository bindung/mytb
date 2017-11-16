import tensorflow as tf


def logical_and(*cond):
    n = len(cond)
    if n < 2:
        raise

    if n == 2:
        return tf.logical_and(*cond)

    c1 = cond[0]
    return tf.logical_and(c1, logical_and(*cond[1:]))

def simple_eval(*arg) :
    with tf.Session() as sess :
        sess.run(tf.global_variables_initializer())
        return sess.run(arg)

if __name__ == "__main__" :
    t = tf.constant(True)
    f = tf.constant(False)

    l = logical_and(f, t)
    print(simple_eval(l))
