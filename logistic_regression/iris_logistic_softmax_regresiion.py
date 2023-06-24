import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris_data = datasets.load_iris()
x = iris_data.get('data')[:, 3:]
y = iris_data.get('target')

# 使用softmax回归
logistic_reg = LogisticRegression(solver='sag', max_iter=1000, multi_class='multinomial')
# 导入数据得到模型
logistic_reg.fit(x, y)

x_new = np.linspace(0, 20, 1000).reshape(-1, 1)
# 在不同0 1 2 的概率
y_predict_proda = logistic_reg.predict_proba(x_new)
# 预测值
y_predict = logistic_reg.predict(x_new)
print(y_predict)