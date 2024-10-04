import tensorflow as tf


def tf_2d_gather(params, idx):
    idx = tf.stack([tf.range(tf.shape(idx)[0]), idx[:, 0]], axis=-1)
    out = tf.gather_nd(params, idx)
    out = tf.expand_dims(out, axis=1)
    return out


if __name__ == '__main__':
    params = tf.constant([[1, 2], [3, 4]])
    idx = tf.constant([[0], [1]])
    out = tf_2d_gather(params, idx)
    print(out.numpy())
