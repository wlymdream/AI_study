from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
class GradientBoostingWithLogisitcRegression:

    def __init__(self):
        self.gbdt_model =None
        self.lr_model = None
        self.gbdt_encoder = None
        self.x_train_leafs = None
        self.x_test_leafs = None
        self.x_trans = None



    ''' 利用gbdt训练模型
        params :
        ----------
        x: 训练集x
        y: 训练接y
    '''
    def gbdt_train_data(self, x, y):
        gbdt_obj = GradientBoostingClassifier(n_estimators=10, max_depth=5, max_features=0.5)
        gbdt_obj.fit(x, y)
        return gbdt_obj

    def lr_train_data(self, x, y):
        lr_obj = LogisticRegression()
        lr_obj.fit(x, y)
        return lr_obj

    def gbdt_lr_train_data(self, x, y):
        # 利用gbdt生成树
        self.gbdt_model = self.gbdt_train_data(x, y)
        # 会返回X这个样本最后落在哪个叶子节点
        self.x_train_leafs = self.gbdt_model.apply(x)[:, :, 0]
        # 对数据进行one-hot编码 0-1编码
        self.gbdt_encoder = OneHotEncoder(categories='auto')
        # 转换成one-hot编码后的数据集
        self.x_trans = self.gbdt_encoder.fit(self.x_train_leafs)
        # 利用逻辑回归
        self.lr_model = self.lr_train_data(self.x_trans, y)
        return self.lr_model

    def gbdt_lr_predict(self, model, x, y):
        self.x_test_leafs = self.gbdt_model.apply(x)[:, :, 0]
        test_rows, cols = self.x_test_leafs.shape
        print(test_rows, cols)
        self.x_trans = self.gbdt_encoder.transform(self.x_test_leafs)
        y_predict = model.predict_proba(self.x_trans)[:, 1]
        auc_scores = roc_auc_score(y, y_predict)
        print('GBDT with LR score : %.5f' % auc_scores)
        return auc_scores
'''
    读取鸢尾花数据集
'''
def load_iris_data():
    iris_datas = load_iris()
    x_data = iris_datas['data']
    # 取y值为0或1 返回的是True False
    y_data = iris_datas['target'] == 2
    return train_test_split(x_data, y_data, test_size=0.4, random_state=0)


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_iris_data()

    gblr_obj = GradientBoostingWithLogisitcRegression()
    gbdt_lr_model_obj = gblr_obj.gbdt_lr_train_data(x_train, y_train)
    gblr_obj.gbdt_lr_predict(gbdt_lr_model_obj, x_test, y_test)

