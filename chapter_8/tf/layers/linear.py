import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense


x = tf.constant([[1.0, 2.0, 3.0]])

linear = Dense(units=2)

# We have to `build` layer to initialize it
linear.build(input_shape=x.shape)

# !!! It doesn't work at all.
# I don't know why, or logical mistake or version problem
# set weights
# linear.set_weights([
#     tf.Variable([[0, 1], [2, 0], [5, 2]]),  # weights
#     tf.Variable([1, 1])  # bias
# ])

# but this works
# set weights
linear.set_weights([
    np.array([[0, 1], [2, 0], [5, 2]]),  # weights
    np.array([1, 1])  # bias
])

y = linear(x)

print(y.numpy())
