# 鸢尾花数据集进行多分类 one-vs-rest  / one-vs-all
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np

iris_data = datasets.load_iris()
x = iris_data.get('data')[:, 3:]
y = iris_data.get('target')

# 创建对象逻辑回归对象
# multi_class=ovr 表示使用one-vs-rest进行多分类 solver= sag使用梯度下降计算模型
logistic_reg_obj = LogisticRegression(solver='sag', max_iter=1000, multi_class='ovr')
logistic_reg_obj.fit(x, y)

x_new = np.linspace(3, 9, 1000).reshape(-1, 1)
y_proba = logistic_reg_obj.predict_proba(x_new)
# print(y_proba)

y_predict = logistic_reg_obj.predict(x_new)
print(y)
