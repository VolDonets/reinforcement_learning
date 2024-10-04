import tensorflow as tf


# Mean square error (MSE) is the most commonly used for regression problems
# mse(y, z) = SUM( (y_i - z_i)^2 ) / n
a = tf.constant([1, 2], dtype=tf.float32)
b = tf.constant([1, 5], dtype=tf.float32)

mse_loss = tf.keras.losses.MeanSquaredError()
mse_error = mse_loss(a, b)

print(f'mse: {mse_error.numpy()}')
