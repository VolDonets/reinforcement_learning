import tensorflow as tf


# absolute loss is the simplest metric of the distance between two vectors:
# absolute_loss(y, z) = SUM( |y_i - z_i| ) / n
a = tf.constant([1, 2], dtype=tf.float32)
b = tf.constant([1, 5], dtype=tf.float32)

abs_loss = tf.keras.losses.MeanAbsoluteError()
abs_error = abs_loss(a, b)

print(f'abs: {abs_error.numpy()}')
