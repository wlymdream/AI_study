# 逻辑回归的损失函数  求最小值，最终的公式前面加了一个负号
# 乳腺癌数据集
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale

data = load_breast_cancer()
# [:, :2]所有行 所有列的前2列
x, y = scale(data['data'][:, :2]), data['target']

# 使用逻辑回归计算模型 不考虑截距项
lr =LogisticRegression(fit_intercept=False)
lr.fit(x, y)
# 模型
theata = lr.coef_
w1 = theata[0, 0]
w2 = theata[0, 1]
print(theata)

# 使用sigmoid函数计算y_hat sigmoid func = 1 / (1 + exp(-theata.T * x))
def get_sigmoid_fun_data(param_x, w1, w2):
    z = param_x[0] * w1 + param_x[1] * w2
    return 1 / (1 + np.exp(-z))

# 逻辑回归损失函数公式-[ y * log(H_theata(x)) +(1-y) * log(1-h_theata(x))] 加和 ->  H_theata(x) = h_hat = 1/(1 + exp(-z))  z = theata.T * X
def logistic_loss_func(param_x, param_y, w1, w2):
    result = 0
    # 遍历每一条样本，计算损失, 结果用result承接  zip()联合到一起变成一个元组，用前面的数据去接
    for single_x, single_y in zip(param_x, param_y):
        # 计算y_hat 预测值
        y_hat = get_sigmoid_fun_data(single_x, w1, w2)
        loss_result = -1 * single_y * np.log(y_hat) - (1 - single_y) * np.log(1 - y_hat)
        result += loss_result
    return result
# 在w1 - 0.6到 w1 + 0.6之间均匀取50个值
w1_space = np.linspace(w1 - 0.6, w1 + 0.6, 50)
w2_space = np.linspace(w2 - 0.6, w2 + 0.6, 50)

# 探索单个参数和损失函数的关系
test_data_w1 = np.array([logistic_loss_func(x, y, i, w2) for i in w1_space])
test_data_w2 = np.array([logistic_loss_func(x, y, w1, i) for i in w2_space])

fig1 = plt.figure(figsize=(8, 6))
# 在2行2列的第一个位置画
plt.subplot(2, 2, 1)
plt.plot(w1_space, test_data_w1)

# 在2行2列的第2个位置画
plt.subplot(2, 2, 2)
plt.plot(w2_space, test_data_w2)

plt.subplot(2, 2, 3)
# w1_space, w2_space 分别作为x,y轴，然后向上向右划线，做网格，网格中的交点为一个个的坐标 他会得到50 * 50个点的坐标
w1_gride, w2_gride = np.meshgrid(w1_space, w2_space)
loss_gride = logistic_loss_func(x, y, w1_gride, w2_gride)
# 等高线图，把一个三维空间的图，用等高线的方式展示出来
plt.contour(w1_gride, w2_gride, loss_gride)

plt.subplot(2, 2, 4)
# 由最后一个图可知，我们要求的最优解在图中心点的位置,就是我们打印的theata的值print(theata)
plt.contour(w1_gride, w2_gride, loss_gride, 30)

fig2 = plt.figure()
ax = Axes3D(fig2, auto_add_to_figure=False)
ax.plot_surface(w1_gride, w2_gride, loss_gride)
fig2.add_axes(ax)
# 在第二张画布上， 我们发现它并不像一个碗，我们想用梯度下降获取到最优解的话，我们可以先把数据进行归一化(归一化是对x进行操作sklearn.preprocessing -> scale)，
# 实现共同富裕，我们去把数据进行优化处理, 经过了scale(data) 归一化处理之后，我们会从3d图上发现它最地点的地方更有弧度了
plt.show()

# 分析x1 和x2两个维度的重要程度
# 在经过归一化之后再去求解模型，从第一张画布上可以发现，前两个图当w1下降0.25之后，
# 第一张图对应的y值改变了1.x左右，第二张图对应的y值改变了2左右的样子，第二张图的变化程度更大， 所以x2要比x1对y值的影响更大


