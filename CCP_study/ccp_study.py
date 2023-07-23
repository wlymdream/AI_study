import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.75)
dtcf = DecisionTreeClassifier()
path = dtcf.cost_complexity_pruning_path(x_train, y_train)
# 得到每个子树的alpha的值
ccp_alphas = path.ccp_alphas
# 得到每个子树的不纯度
impurities = path.impurities

clfs = []
for single_ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(ccp_alpha=single_ccp_alpha)
    clf.fit(x_train, y_train)
    clfs.append(clf)

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]
nons_count = [clf.tree_.node_count for clf in clfs]
depths = [clf.tree_.max_depth for clf in clfs]

plt.plot(ccp_alphas, nons_count, 'r-')
plt.plot(ccp_alphas, depths, 'b-')

plt.show()

train_scores= [clf.score(x_train, y_train) for clf in clfs]
tests_scores = [clf.score(x_test, y_test) for clf in clfs]

plt.plot(ccp_alphas, train_scores, label='train', marker='o', drawstyle='steps-post')
plt.plot(ccp_alphas, tests_scores, label='test', marker='o', drawstyle='steps-post')
plt.legend()
plt.show()

