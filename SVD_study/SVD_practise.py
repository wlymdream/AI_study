# 奇异值分解 SVD可以是PCA的一种解法
# PCA算法步骤：生成协方差矩阵 获取特征值和特征向量 对应有最大的几个特征值的特征向量被用于重新建立数据，重建的数据包含原始数据的大部分方差
# PCA的解法有两种 1 特征值分解 A.T * A = P * K * P^-1 = P * K * P.T P是一堆特征向量 K是一堆特征值
# 2 SVD 奇异值分解 A = U * E * V.T  A.T = V * E.T * U.T  A.T * A = V * E^2 * V.T  A.T * A为协方差矩阵
# 由于U * U.T是一个单位阵，所以我们不写了 得到上面的式子 由此可知 V 为一堆的特征向量 E^2 为一堆特征值

import numpy as np

'''
    Args : 
        m:  (m * n) 的矩阵
    Returns:
        s: m * m 的矩阵
        v: m * n 的矩阵
        d: n * n 的矩阵
'''
def get_matrix_by_svd(m):
    s, v, d = np.linalg.svd(m)
    result = {'s': s, 'v': v, 'd':d}
    return result

M =[[1,2,3], [4,5,6], [7,8,9], [10, 11, 12], [13, 14, 15],[16,16,18]]
results = get_matrix_by_svd(M)
print(results)
