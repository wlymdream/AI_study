import sklearn.datasets
import tensorflow as tf
import numpy as np
import os
from sklearn.datasets import fetch_california_housing
tf.compat.v1.disable_eager_execution()

# 加载数据 如果该目录下没有文件就自动下载 我下载的时候报错了…… 我放弃…… 我手动下载的地址https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.tgz）
housing_data = fetch_california_housing()
m, n = housing_data.data.shape
# print(m, n)
# print(housing_data.feature_names)
# print(housing_data.data)
# print(housing_data.target)

# 添加截距项
housing_data_bias = np.c_[np.ones((m, 1)), housing_data.data]
# print(housing_data_bias)
# 获取x和y  constant常量节点 y给转换一下，变成m行1列

x = tf.constant(housing_data_bias, dtype=tf.float32)
y = tf.constant(housing_data.target.reshape(-1, 1), dtype=tf.float32)
# x的转置
xt = tf.transpose(x)
# 用tf框架提供的矩阵操作求theta 用解析解的方式求theata (x.T * x)^-1 * x.T * y
theata = tf.matmul(tf.matmul(tf.linalg.inv(tf.matmul(xt, x)), xt), y)

with tf.compat.v1.Session() as sess:
    # print(theata)
    theata_values = sess.run(theata)
print(theata_values)




