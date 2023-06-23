# 逻辑回归：逻辑回归是一个分类算法，它是基于多元线性回归的，逻辑回归这个分类算法是线性的分类器，
# sigmoid funcion 是一个s型曲线，他的公式是 y = 1 / 1 + e^-z 当z为0的时候 y的值为0.5
import numpy as np
import math as mt
import matplotlib.pyplot as plt

def sigmoid_regression_func(x):
    a = []
    for single_data in x:
        # mt.exp(-single_data)  e的负single_data次方
        a.append(1.0/(1.0 + mt.exp(-single_data)))
    return a

# arange 生成等差数列 -10到10之间每隔0.1取一个数
x = np.arange(-10, 10, 0.5)
y = sigmoid_regression_func(x)

plt.plot(x, y)
plt.show()
