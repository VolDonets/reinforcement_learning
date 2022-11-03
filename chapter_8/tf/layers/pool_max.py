import tensorflow as tf
from keras.layers import MaxPool2D, AvgPool2D


A = tf.constant(
    [[
        [[1], [2], [-1], [1]],
        [[0], [1], [-2], [-1]],
        [[3], [0], [5], [0]],
        [[0], [1], [4], [-3]]
    ]], dtype='float')

max_pool = MaxPool2D(pool_size=2)
avg_pool = AvgPool2D(pool_size=2)

out = max_pool(A)
print(out.numpy())

# !!! For AvgPool I should set A's type as float.
avg_out = avg_pool(A)
print()
print(avg_out.numpy())
