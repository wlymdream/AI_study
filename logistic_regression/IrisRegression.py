# 对鸢尾花数据集进行二分类
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

iris_data = datasets.load_iris()
# print(iris_data.keys())
# print(iris_data.get('DESCR'))
# print(iris_data.get('feature_names'))

# 取data中所有行，取第3列数据
x = iris_data['data'][:, 3:]
# print(x)

# 由于我们最终拿到的y值是3中类型，我们把它变成二分类
y = (iris_data['target'] == 2).astype(int)
# print(y)

# sag 随机梯度下降 max_iter 最多迭代次数
logistic_obj = LogisticRegression(solver='sag', max_iter=1000)
logistic_obj.fit(x, y)
# 在0到3之间均匀取1000个值, 变成列向量
x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
print(x_new)

# 得到的值第一列为负例的概率，第二列为正例的概率
y_proda = logistic_obj.predict_proba(x_new)

# 预测值
y_hat = logistic_obj.predict(x_new)
print(y_hat)