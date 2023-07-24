# 利用随机森林对鸢尾花数据集分类
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:, :]
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# n_estimators 多少棵小树构成一个森林 max_leaf_nodes max_leaf_nodes最多多少个叶子结点 n_jobs并行度
rnd_clf_obj = RandomForestClassifier(n_estimators=15, max_leaf_nodes=16, n_jobs=1)
rnd_clf_obj.fit(x_train, y_train)

y_test_predict = rnd_clf_obj.predict(x_test)
accuracy_scores = accuracy_score(y_test, y_test_predict)
print(accuracy_scores)

feature_importances = rnd_clf_obj.feature_importances_
print(feature_importances)

for name, score in zip(iris.feature_names, feature_importances):
    print(name, score)

log_obj = LogisticRegression()
svc_obj = SVC()
dt_clf_obj = DecisionTreeClassifier()
bagging_obj = BaggingClassifier(DecisionTreeClassifier(), n_estimators=10, n_jobs=1, bootstrap=True)
bagging_obj.fit(x_train, y_train)
y_bagging_test_predict = bagging_obj.predict(x_test)
bag_score = bagging_obj.score(x_test, y_test)
print("====", bag_score)

vt_clf_obj = VotingClassifier(estimators=[('lr', log_obj), ('svc', svc_obj), ('dt_clf', dt_clf_obj)])
vt_clf_obj.fit(x_train, y_train)
y_predict = vt_clf_obj.predict(x_test)
scores = vt_clf_obj.score(x_test, y_test)
print("====", scores)

