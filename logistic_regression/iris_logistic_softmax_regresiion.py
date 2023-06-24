import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris_data = datasets.load_iris()
x = iris_data.get('data')[:, 3:]
y = iris_data.get('target')

# 使用softmax回归
# 当multi_class的值为ovr的时候，是使用的逻辑回归 他是会拆分成3个二分类算法，每个二分类算法之间是相互独立的
# 当使用softmax回归的时候，它是直接使用多项式分类，他不管再怎么计算的时候，它的分母都是一样的，它更倾向于把结果算在一个类别，尽可能的把相应的概率算的大一些
# 然而ovr的1个类别他是算在3个二分类中的，softmax会有一个概率之间的相互抑制，而ovr不会，它会拆分，然后个算各的
# 当我们要计算的数据必须要有明确的分类，我们使用softmax更好一些，但是如果计算的数据是属于一个数据属于哪几种分类，我们使用ovr更好一些
logistic_reg = LogisticRegression(solver='sag', max_iter=1000, multi_class='multinomial')
# 导入数据得到模型
logistic_reg.fit(x, y)

x_new = np.linspace(0, 20, 1000).reshape(-1, 1)
# 在不同0 1 2 的概率
y_predict_proda = logistic_reg.predict_proba(x_new)
# 预测值
y_predict = logistic_reg.predict(x_new)
print(y_predict)