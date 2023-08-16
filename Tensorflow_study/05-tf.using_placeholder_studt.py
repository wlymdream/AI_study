import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# shape中的None 表示可以任意行 但必须是3列
a = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 3))
b = a + 5

with tf.compat.v1.Session() as sess:
    b_val_1 = b.eval(feed_dict = {a:[[1,2,3]]})
    b_val_2 = b.eval()

