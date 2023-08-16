import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
tf.compat.v1.disable_eager_execution()

#1 加载数据
housing_data = fetch_california_housing()
# 求数据有多少行 多少列
m, n = housing_data.data.shape

x_bias = np.c_[np.ones((m, 1)), housing_data.data]
y_data = housing_data.target

# 对数据进行归一化处理 使用方差归一化和均值归一化同时进行
scaler_obj = StandardScaler(with_std=True, with_mean=True)
scaler_obj.fit(x_bias)
# 得到归一化后的x
scaler_x_bias = scaler_obj.transform(x_bias)

x = tf.constant(scaler_x_bias, dtype=tf.float32)
y = tf.constant(y_data.reshape(-1, 1), dtype=tf.float32)

# 2. 初始化theata
theta = tf.Variable(tf.random.uniform([n + 1, 1], -1, 1))
print(theta)
# 3. 求梯度 损失函数 (y_hat - y)^2再求均值  g = (h(thetax) - y) * x    h(thetax) = y_hat  y_hat = x*theta error = y_hat -y
with tf.GradientTape() as tape:
    # tape.watch(theta)
    y_predict = tf.matmul(x, theta)
    print(y_predict)
    error = y_predict - y
    print('Error', error)
    mse_loss = tf.reduce_mean(tf.square(error))
    rmse = tf.sqrt(mse_loss)
# 对谁求导，第二个参数就写谁
g = tape.gradient(mse_loss, theta)
print(g)
# print('MES', mse_loss)
# rmse = tf.sqrt(mse_loss)
# g = 2/m * tf.matmul(tf.transpose(x), error)

learning_rate = 0.001
epoches = 36500
# 4 更新theta
training_op = tf.compat.v1.assign(theta, theta - learning_rate * g)

init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init)
    sess.run(g)
    for epoch in range(epoches):
        if epoch % 100 == 0:
            # rmse。eval() 损失评分
            print('Epoch', epoch, 'RME:', rmse.eval())
        sess.run(training_op)
    best_thea = theta.eval()
print(best_thea)






