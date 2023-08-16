import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
# 主要是用于做标准归一化
from sklearn.preprocessing import StandardScaler

tf.compat.v1.disable_eager_execution()

learning_rate = 0.001
n_epochs = 36500

housing_data = fetch_california_housing()
m, n = housing_data.data.shape

x_bias_data = np.c_[np.ones((m, 1)), housing_data.data]
y_data = housing_data.target

# 默认就是既用均值归一化也用方差归一化
scaler = StandardScaler(with_mean=True, with_std=True)
scaler.fit(x_bias_data)
# 进行归一化（归一化针对的是x）
scaler_x_plus_bias = scaler.transform(x_bias_data)

x = tf.constant(scaler_x_plus_bias, dtype=tf.float32)
y = tf.constant(y_data.reshape(-1, 1), dtype=tf.float32)

# def make_variables(k, initalizer):
#     return tf.Variable(initalizer(shape=k))

# 进行梯度下降
# 初始化w 均匀分布 如果不指定取值是0-1之间 初始化一个n+1行 1列 tf.random_normal_initializer服从标准正态分布
#  我们让w趋近于0可以想到L1 L2正则项 我让他从0开始调
theta = tf.Variable(tf.random.uniform([n+ 1, 1], -1, 1))
# 求梯度 求梯度对loss求偏导 loss是根据y_hat和真实的y来的，y_hat是把x带入计算出来额theta中 y_hat = x * theta
print(theta)
y_pred = tf.matmul(tf.cast(x, tf.float32), theta)
# loss
error = y_pred - tf.cast(y, tf.float32)
rmse = tf.sqrt(tf.reduce_mean(tf.transpose(error)))
# MSE的导函数的公式 (y_hat - y) * X -> X.T * (y_hat -y)
gradients = 2/m * tf.matmul(tf.transpose(x), error)
# 更新theta theata(t +1) = theata(t) - learning_rate * g g梯度
training_op = tf.compat.v1.assign(theta, theta - learning_rate * gradients)

init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:

    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print('Epoch', epoch, 'RMES=', rmse.eval())

        sess.run(training_op)
    best_theata = theta.eval()
    print(best_theata)



