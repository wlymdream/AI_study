# KMeans聚类
import numpy as np
# 分词器
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

def distCos(vecA, vecB):

    return np.dot(vecA, vecB) / (np.sqrt(np.sum(np.square(vecA))) * np.sqrt(np.sum(np.square(vecB))))

def distEclud(vecA, vecB):
    # 欧氏距离公式 (a-b)^2加和 然后开根号
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

def randCent(data_param, k):
    m, n = data_param.shape[0], data_param.shape[1]
    # 获取索引列表
    index_ist = list(range(m))
    # 打乱顺序
    np.random.shuffle(index_ist)
    # 创建中心点
    result = data_param[index_ist][k]
    return result

def KMeans_func(data_param, k, disMeans=distEclud, createdCenter=randCent):
    x = data_param.shape[0]
    # 每个样本属于哪个类别，以及到中心点的距离
    centerDistance = np.zeros((x, 2))
    # 1. 初始化k个中心点
    center = createdCenter(data_param, k)
    clusterChangedStatus = True
    while clusterChangedStatus:
        clusterChangedStatus = False
        # 2. 把每个数据点划分到离它最近的中心点的所属类别 EM（期望最大化）
        # 期望最大化算法分成2步，E -> 划分  M -> 更新
        # 遍历每个样本
        for i in range(x):
            # 把距离设置为正无穷
            minDist = float('inf')
            # 设置最小索引
            minIndex = -1
            # 遍历每个簇
            for j in range(k):
                # 求样本点到中心点的距离（欧氏距离）
                disJ = disMeans(center[j, :], data_param[i, :])
                # 如果算出来的距离小于我们设定的最小距离
                if disJ < minDist:
                    minDist = disJ
                    minIndex = j
            # 如果本次的簇和之前划分的簇不一样
            if centerDistance[i, 0] != minIndex:
                clusterChangedStatus = True
            # 把划分的样本记录下来
            centerDistance[i, :] = minIndex, minDist **2
            print(center)
        # 3. 更新u1到uk 重新计算中心点的坐标
        for center in range(k):
            # 划分到数据集里的样本取出来
            ptsInClust = data_param[np.nonzero(centerDistance[:, 0] == center)]
            # 把簇里的样本点求平均
            centerDistance[center, :] = np.mean(ptsInClust, axis=0)
    return center, centerDistance

if __name__ == '__main__':
    text = '我爱北京天安门'
    text2 = '我爱北京颐和园'
    text3 = '欧洲杯意大利夺冠了'
    textVectors = [text, text2, text3]
    textVectors = [' '.join(list(jieba.cut(doc))) for doc in textVectors]
    print(textVectors)
    X = TfidfVectorizer().fit_transform(textVectors)
    # print("====", X)
    print(X.A)
    # # X.A 相当于np.array(X)
    # result = KMeans_func(X.A, k=2, disMeans=distCos)
    # print(result)

