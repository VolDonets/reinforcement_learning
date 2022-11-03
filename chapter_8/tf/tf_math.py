import tensorflow as tf

x = tf.ones((1, 2), dtype=tf.float32)  # (1, 1)
print('x:', x)

y = tf.range(0, 2, dtype=tf.float32)  # (0, 1)
print('y:', y, '\n')

# implicit addition
z = x + y  # (1, 2)
print('x + y = z =', z, '\n')

# explicit addition
w = tf.add(z, y)  # (1, 3)
print('z + y = w =', w, '\n')

# implicit multiplication
k = w * -1  # (-1, -3)
print('w * -1 = k =', k, '\n')

# absolute value
a = tf.abs(k)  # (1, 3)
print('|k| = a =', a, '\n')

# implicit division
b = a / 2  # (0.5, 1.5)
print('a / 2 = b =', b, '\n')

# Rounding to the nearest integer lower than
c = tf.floor(b)  # (0, 1)
print('floor(b) = c =', c, '\n')

# Rounding to the nearest integer greater than
d = tf.math.ceil(b)  # (1, 2)
print('ceil(b) = d =', d, '\n')

# Computes element-wise equality
eq = tf.equal(c, d)  # (False, False)
print('equal(c, d) = eq =', eq, '\n')

# Mean tensor value
avg = tf.reduce_mean(d)  # 1.5
print('reduce_mean(d) = avg =', avg, '\n')

# Max tensor value
mx = tf.math.reduce_max(d)  # 2
print('reduce_max(d) = mx =', mx, '\n')

# Min tensor value
mn = tf.math.reduce_min(d)  # 1
print('reduce_min(d) = mn =', mn, '\n')

# Sum of all tensor values
sm = tf.math.reduce_sum(d)  # 3
print('reduce_sum(d) = sm =', sm, '\n')
