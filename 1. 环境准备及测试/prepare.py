import tensorflow as tf

tf.compat.v1.disable_eager_execution()

A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])

# 矩阵乘法
C = tf.matmul(A, B)
print(C)

#  tf.GradientTape() 计算函数 y(x) = x**2 在 x = 3 时的导数

x = tf.compat.v1.get_variable('x', shape=[1], initializer=tf.constant_initializer(3.))
with tf.GradientTape() as tape:
    y = tf.square(x)
# 计算 y 关于 x 的导数
    y_grad = tape.gradient(y, x)
    print([y.numpy(), y_grad.numpy()])
