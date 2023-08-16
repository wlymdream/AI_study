import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
tf.compat.v1.disable_eager_execution()

n_epoches = 1000
learning_rate = 0.001
batch_size = 2000

# 1.加载数据
housing_data = fetch_california_housing()
m, n = housing_data.data.shape

x_train, x_test, y_train, y_test = train_test_split(housing_data.data, housing_data.target)
scaler = StandardScaler()
# fit 是在训练均值和标准差
scaler.fit(x_train)
# （xij - 这一列的均值）/ 这一列的标准差
x_train_scaler = scaler.transform(x_train)
x_train_data = np.c_[np.ones((len(x_train_scaler), 1)), x_train_scaler]
# x_test不做fit是因为我们假设训练集和测试集是同分布的
x_test_scaler = scaler.transform(x_test)
x_test_data = np.c_[np.ones((len(x_test_scaler), 1)), x_test_scaler]

# 占位符 什么时候真正去跑的时候，才会真正的往里面填数据
x = tf.compat.v1.placeholder(dtype=tf.float32)
y = tf.compat.v1.placeholder(dtype=tf.float32)

theta = tf.Variable(tf.random.uniform([n+1, 1], -1, 1))
y_predict = tf.matmul(x, theta)
error = y_predict - y
mse = tf.reduce_mean((tf.square(error)))
# 传入一个mse 他会自己求梯度 对谁求导取决于
training_op = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(mse)
init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    n_batch = int(len(x_train_data) / batch_size)
    for epoch in range(n_epoches):
        temp_theta = theta.eval()
        if epoch% 100 == 0:
            print(temp_theta)

            print('Epoch:', epoch, 'MSE:', sess.run(mse, feed_dict={
                x:x_train_data,
                y:y_train
            }))

            arr = np.arange(len(x_train_data))
            np.random.shuffle(arr)
            x_train_data = x_train_data[arr]
            y_train = y_train[arr]

            for i in range(n_batch):
                sess.run(training_op, feed_dict={
                    x:x_train_data[i * batch_size: i*batch_size + batch_size],
                    y:y_train[i*batch_size: i*batch_size + batch_size]
                })

        best_theta = theta.eval()
print(best_theta)




