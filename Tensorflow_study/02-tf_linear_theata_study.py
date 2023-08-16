import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# 加载数据
housing_data = fetch_california_housing()
# 获取行数和列数
m, n = housing_data.data.shape

x_data = housing_data.data
y_data = housing_data.target

x_data_bias = np.c_[np.ones((m, 1)), x_data]

x = tf.constant(x_data_bias, dtype=tf.float32)
y = tf.constant(y_data.reshape(-1, 1), dtype=tf.float32)

xt = tf.transpose(x)

theata = tf.matmul(tf.matmul(tf.linalg.inv(tf.matmul(xt, x)), xt), y)

with tf.compat.v1.Session() as sess:
    theata_values = sess.run(theata)
print(theata_values)
