#-*- coding: utf-8 -*-
# Author: Bob
# Date:   2016.11.24
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from scipy import linalg
iris = load_iris()
yc = iris.target

Xc = pd.read_csv('../data/Mnist/mnist_train.csv',header = None,nrows =500)
print(Xc.head())

# Xc = iris.data
Xc=Xc/255.
k=3
pcac = PCA(n_components=k,whiten=False) #实例化
pcac = pcac.fit(Xc) #拟合模型
Xc_dr = pcac.transform(Xc) #获取新矩阵
Xc_dr.shape
# print(pcac.get_covariance())
# [[ 0.67919741 -0.03258618  1.27066452  0.5321852 ]
#  [-0.03258618  0.18113034 -0.31863564 -0.13363564]
#  [ 1.27066452 -0.31863564  3.11934547  1.28541527]
#  [ 0.5321852  -0.13363564  1.28541527  0.58961806]]
print(pcac.explained_variance_)
# [4.22484077 0.24224357]
print(pcac.explained_variance_ratio_)
# [0.92461621 0.05301557]
Vc = pcac.components_
print(Vc)
# [[ 0.36158968 -0.08226889  0.85657211  0.35884393]
#  [ 0.65653988  0.72971237 -0.1757674  -0.07470647]]
print('*'*30)

### 只减去均值
print('second')
mean_ = np.mean(Xc, axis=0)
std_ = np.std(Xc,axis=0)
# Xc1 = (Xc-mean_)/(std_+10**-7)
Xc1 = Xc-mean_
U, S, V = linalg.svd(Xc1)

V[0,:]=V[0,:]*(-1)
# V[1,:]=V[1,:]*(-1)

# print(S)
print(V[:k,:])
# [[ 0.36158968 -0.08226889  0.85657211  0.35884393]
#  [ 0.65653988  0.72971237 -0.1757674  -0.07470647]]
Xc_dr1 = Xc1.dot(V[:k,:].T)
# print(Xc_dr-Xc_dr1)

Xc_dr11 = Xc1.dot(Vc.T)
print((Xc_dr-Xc_dr1).sum(axis=1))

# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0.]

### 只减去均值
# print('third')
# cov = Xc1.T.dot(Xc1)/149
# # print(cov)
# # [[ 0.68569351 -0.03926846  1.27368233  0.5169038 ]
# #  [-0.03926846  0.18800403 -0.32171275 -0.11798121]
# #  [ 1.27368233 -0.32171275  3.11317942  1.29638747]
# #  [ 0.5169038  -0.11798121  1.29638747  0.58241432]]
# u,d,v = np.linalg.svd(cov)
# # d:[4.22484077 0.24224357 0.07852391 0.02368303]
# u = -1*u
# print(u[:,:k])
# Xc_dr2 = np.dot(Xc1, u[:,:k])
# print((Xc_dr-Xc_dr2).sum(axis=1))

### 不使用SVD
# U,V = np.linalg.eigh(cov) 
# U = U[::-1]
# # [4.22484077 0.24224357 0.07852391 0.02368303]
# for i in range(4):
#     V[i,:] = V[i,:][::-1]
# v = V[:,:k]
# v[:,0]=-1*v[:,0]
# # print(v)
# # [[ 0.36158968  0.65653988]
# #  [-0.08226889  0.72971237]
# #  [ 0.85657211 -0.1757674 ]
# #  [ 0.35884393 -0.07470647]]

# Xc_dr3 = np.dot(Xc1, v)
# print((Xc_dr-Xc_dr3).sum(axis=1))
# ### 除以标准差
# print('fourth')
# n = Xc.shape[1]
# std_ = np.std(Xc,axis=0)
# print(std_)
# Xc2 = (Xc-mean_)/std_
# print(Xc2.mean())
# print(Xc2.std())
# U2, S2, V2 = linalg.svd(Xc2)
# print(V2)
# # , full_matrices=False
# V2[1,:]=V2[1,:]*(-1)
# # print(S2)
# Xc_dr2 = Xc2.dot(V2[:k,:].T)
# print(V2[:k,:])
# print((Xc_dr-Xc_dr2).sum(axis=1))

