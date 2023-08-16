import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

x = np.array([[0, 0], [1, 1]])
y = np.array([0, 1])
print(x.shape, y.shape)

# activation 激活函数 一般用在隐藏层 输出层一般会用一些非线性变换sigmoid或者是softmax
# tol 忍受度 最后两次loss的差值的绝对值小于10^-4 就停止迭代 连续10次会停止迭代
# verbose在训练的过程中要不要多打印一些信息
# hidden_layer_sizes 表示有2个隐藏层 第一个隐藏层有5个节点，第二个隐藏层有2个节点
mlp_clf_obj = MLPClassifier(
    solver='sgd',
    alpha=1e-5,
    activation='relu',
    hidden_layer_sizes=(5,2),
    max_iter=2000,
    tol=1e-4,
    verbose=True
)

mlp_clf_obj.fit(x, y)

predicts = mlp_clf_obj.predict([[2,2], [-1, -2]])
print(predicts)

predicts_proda = mlp_clf_obj.predict_proba([[2,2], [-1, -2]])
print(predicts_proda)

print([coef.shape for coef in mlp_clf_obj.coefs_])
print([coef for coef in mlp_clf_obj.coefs_])

print([coef.shape for coef in mlp_clf_obj.intercepts_])
print([coef for coef in mlp_clf_obj.intercepts_])
