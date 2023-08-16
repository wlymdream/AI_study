import tensorflow as tf
tf.compat.v1.disable_eager_execution()

print(tf.__version__)
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train)

x = tf.Variable(3)
print(x)
y = tf.Variable(4)
print(y)

f = x*x*y + y + 2
print(f)
# 上面属于构建一个graph图
# 下面属于构建一个运行graph图
# 创建一个运行图
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
sess.run(x.initializer)
sess.run(y.initializer)
# print(sess.run(f))
sess.close()

x1 = tf.Variable(1)
# print(x1.graph)
graph = tf.Graph()
x3 = tf.Variable(3)
with graph.as_default():
    x2 = tf.Variable(2)

x4 = tf.Variable(3)
# print(x1.graph is graph)
# print(x2.graph is graph)
# print(x4.graph is graph)
# print(x3.graph is graph)

# 当去计算一个节点的时候，Tensorflow会自动计算它一来的一组节点的值，并且首先计算依赖节点的值
w = tf.Variable(5)
x5 = w + 2
y = x5 + 5
z = x5 * 3

with tf.compat.v1.Session() as session:
    session.run(w.initializer)
    print(session.run(y))
    # 这里为了计算z，又重新计算了x和w，除了Variable值，tf不会缓存其他的值，比如contant的值
    # 一个Variable的生命周期是当它的initialer运行的时候，到会话seesion close的结束
    print(session.run(z))

with tf.compat.v1.Session() as session:
    session.run(w.initializer)
    y_val, z_val = session.run([y, z])
    print(y_val)
    print(z_val)