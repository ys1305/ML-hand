#导入相应包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None,aweights=None)

# m:一维或则二维的数组，默认情况下每一行代表一个变量（属性），每一列代表一个观测样本
# y:与m具有一样的形式的一组数据
# rowvar:默认为True,此时每一行代表一个变量（属性），每一列代表一个观测；为False时，则反之
# bias:默认为False,此时标准化时除以n-1；反之为n。其中n为观测数
# ddof:类型是int，当其值非None时，bias参数作用将失效。当ddof=1时，将会返回无偏估计（除以n-1），
# 即使指定了fweights和aweights参数；当ddof=0时，则返回简单平均值。
# frequency weights:一维数组，代表每个观测要重复的次数（相当于给观测赋予权重）
# analytic weights:一维数组，代表观测矢量权重。对于被认为“重要”的观察,
# 这些相对权重通常很大,而对于被认为不太重要的观察,这些相对权重较小。如果ddof = 0,则可以使用权重数组将概率分配给观测向量。


#导入数据
testSet = pd.read_table('testSet.txt',header=None)
dataSet = testSet
#计算均值
meanVals = dataSet.mean(0)
#去均值化，均值变为0 
meanRemoved = dataSet - meanVals
#计算协方差矩阵
covMat = np.mat(np.cov(meanRemoved, rowvar=0))

print(covMat)
# print(np.cov(meanRemoved))
print(np.cov(meanRemoved, rowvar=0))
print(np.cov(meanRemoved, rowvar=0,bias=1))
print(meanRemoved.T.dot(meanRemoved)/(len(meanRemoved)-1))

def pca(dataSet, N=9999999):
    meanVals = dataSet.mean(0)                      
    meanRemoved = dataSet - meanVals               
    covMat = np.mat(np.cov(meanRemoved, rowvar=0))  
    eigVals,eigVects = np.linalg.eig(covMat)
    # 对特征值排序，.argsort()函数默认从小到大排序，返回的是索引      
    eigValInd = np.argsort(eigVals)

    # 提取出最大的N个特征
    eigValInd = eigValInd[:-(N+1):-1]

    redEigVects = eigVects[:,eigValInd]

    # 降维后的数据
    lowDDataMat = np.mat(meanRemoved) * redEigVects

    # 降维数据重构为原来数据
    reconMat = (lowDDataMat * redEigVects.T) + np.mat(meanVals)
    return lowDDataMat, reconMat

lowDDataMat, reconMat = pca(testSet, N=1)

from sklearn.decomposition import PCA
pcac = PCA(n_components=1)
pcac = pcac.fit(testSet) #拟合模型
Xc_dr = pcac.transform(testSet)
print(Xc_dr.shape)
pcaniv = pcac.inverse_transform(Xc_dr)

print(reconMat.shape)
print((reconMat-pcaniv).sum())
print(pcaniv[:,0].shape) # (1000,)
print(reconMat[:,0].A.flatten().shape) # (1000,)
print(reconMat[:,0].shape) # (1000, 1) 不能进行绘图
print(reconMat[:,0].flatten().shape) # (1, 1000)
print(reconMat[:,0].A.shape) # (1000, 1)
plt.scatter(testSet.iloc[:,0],testSet.iloc[:,1],marker = '.',c='orange')
plt.scatter(reconMat[:,0].A.flatten(),reconMat[:,1].A.flatten(), marker='*',c='g')
plt.scatter(pcaniv[:,0],pcaniv[:,1],marker = '.',c='b',s=3)
plt.show()