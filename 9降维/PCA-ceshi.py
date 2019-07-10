#-*- coding: utf-8 -*-
# Author: Bob
# Date:   2016.11.24
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from scipy import linalg
iris = load_iris()
yc = iris.target
Xc = iris.data

print(Xc.mean(axis=0))
print(Xc.std(axis=0))
# print(np.cov(Xc.T)) 
# 等于减去均值后的cov
# [[ 0.68569351 -0.03926846  1.27368233  0.5169038 ]
#  [-0.03926846  0.18800403 -0.32171275 -0.11798121]
#  [ 1.27368233 -0.32171275  3.11317942  1.29638747]
#  [ 0.5169038  -0.11798121  1.29638747  0.58241432]]
k=2
pcac = PCA(n_components=k) #实例化
pcac = pcac.fit(Xc) #拟合模型
Xc_dr = pcac.transform(Xc) #获取新矩阵
Xc_dr.shape
print(pcac.get_covariance())
# 不是简单的返回样本的协方差,只有当k=4时返回的才是样本的协方差，源代码见下面
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

# print(Xc_dr-(Xc-Xc.mean(axis=0)).dot(Vc.T))
print('*'*30)

### 只减去均值
print('second')
mean_ = np.mean(Xc, axis=0)
Xc1 = Xc-mean_
U, S, VT = linalg.svd(Xc1)
unp,snp,vnp = np.linalg.svd(Xc1)
print(unp.shape)
#(150, 150)
print(snp.shape)
#(4,)
print(snp)
# [25.08986398  6.00785254  3.42053538  1.87850234]
print(S)
# [25.08986398  6.00785254  3.42053538  1.87850234]

print(VT)
# [[ 0.36158968 -0.08226889  0.85657211  0.35884393]
#  [-0.65653988 -0.72971237  0.1757674   0.07470647]
#  [ 0.58099728 -0.59641809 -0.07252408 -0.54906091]
#  [ 0.31725455 -0.32409435 -0.47971899  0.75112056]]

# 协方差矩阵的特征值
explained_variance_ = (S ** 2) / (150 - 1)
# print(explained_variance_)
# # [4.22484077 0.24224357 0.07852391 0.02368303]
# total_var = explained_variance_.sum()
# explained_variance_ratio_ = explained_variance_ / total_var
# print(explained_variance_ratio_)
# # [0.92461621 0.05301557 0.01718514 0.00518309]

VT[1,:]=VT[1,:]*(-1)
VT[2,:]=VT[2,:]*(-1)
# print(S)
# print(VT[:k,:])
# [[ 0.36158968 -0.08226889  0.85657211  0.35884393]
#  [ 0.65653988  0.72971237 -0.1757674  -0.07470647]]
Xc_dr1 = Xc1.dot(VT[:k,:].T)
# print(Xc_dr-Xc_dr1)
# print((Xc_dr-Xc_dr1).sum(axis=1))
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0.]

### 只减去均值
print('third')
cov = Xc1.T.dot(Xc1)/149
# print(np.cov(Xc1.T))
# [[ 0.68569351 -0.03926846  1.27368233  0.5169038 ]
#  [-0.03926846  0.18800403 -0.32171275 -0.11798121]
#  [ 1.27368233 -0.32171275  3.11317942  1.29638747]
#  [ 0.5169038  -0.11798121  1.29638747  0.58241432]]
# print(cov)
# [[ 0.68569351 -0.03926846  1.27368233  0.5169038 ]
#  [-0.03926846  0.18800403 -0.32171275 -0.11798121]
#  [ 1.27368233 -0.32171275  3.11317942  1.29638747]
#  [ 0.5169038  -0.11798121  1.29638747  0.58241432]]
u,d,v = np.linalg.svd(cov)

# print((u-v.T))
# [[ 2.77555756e-16  6.66133815e-16  6.66133815e-16  1.11022302e-16]
#  [ 2.77555756e-17 -7.77156117e-16  0.00000000e+00 -5.55111512e-17]
#  [-1.11022302e-16  5.55111512e-17 -1.11022302e-16  5.55111512e-17]
#  [-5.55111512e-17 -1.11022302e-16  0.00000000e+00  1.11022302e-16]]

# print(u)
# [[-0.36158968 -0.65653988  0.58099728  0.31725455]
#  [ 0.08226889 -0.72971237 -0.59641809 -0.32409435]
#  [-0.85657211  0.1757674  -0.07252408 -0.47971899]
#  [-0.35884393  0.07470647 -0.54906091  0.75112056]]
# d:[4.22484077 0.24224357 0.07852391 0.02368303]
u = -1*u
print(u[:,:k])
Xc_dr2 = np.dot(Xc1, u[:,:k])
# print((Xc_dr-Xc_dr2).sum(axis=1))

### 不使用SVD
U,V = np.linalg.eigh(cov) 
U = U[::-1]
print(U)
# [4.22484077 0.24224357 0.07852391 0.02368303]
for i in range(4):
    V[i,:] = V[i,:][::-1]
v = V[:,:k]
v[:,0]=-1*v[:,0]
# print(v)
# [[ 0.36158968  0.65653988]
#  [-0.08226889  0.72971237]
#  [ 0.85657211 -0.1757674 ]
#  [ 0.35884393 -0.07470647]]

print('未除以样本数减1')
U,V = np.linalg.eigh(Xc1.T.dot(Xc1)) 
print(U)
print(V)
print('################')
Xc_dr3 = np.dot(Xc1, v)
# print((Xc_dr-Xc_dr3).sum(axis=1))
# ### 除以标准差
print('fourth')
n = Xc.shape[1]
std_ = np.std(Xc,axis=0)
print(std_)
Xc2 = (Xc-mean_)/std_
print(Xc2.mean())
print(Xc2.std())
U2, S2, V2 = linalg.svd(Xc2)
print(V2)
# , full_matrices=False
V2[1,:]=V2[1,:]*(-1)
# print(S2)
Xc_dr2 = Xc2.dot(V2[:k,:].T)
# print(V2[:k,:])
# print((Xc_dr-Xc_dr2).sum(axis=1))


# def get_covariance(self):
#     """Compute data covariance with the generative model.
#     ``cov = components_.T * S**2 * components_ + sigma2 * eye(n_features)``
#     where S**2 contains the explained variances, and sigma2 contains the
#     noise variances.
#     Returns
#     -------
#     cov : array, shape=(n_features, n_features)
#         Estimated covariance of data.
#     """
#     components_ = self.components_
#     exp_var = self.explained_variance_
#     if self.whiten:
#         components_ = components_ * np.sqrt(exp_var[:, np.newaxis])
#     exp_var_diff = np.maximum(exp_var - self.noise_variance_, 0.)
#     cov = np.dot(components_.T * exp_var_diff, components_)
#     cov.flat[::len(cov) + 1] += self.noise_variance_  # modify diag inplace
#     return cov

noise_variance_ = explained_variance_[2:].mean()
components_ = pcac.components_
exp_var = pcac.explained_variance_
exp_var_diff = np.maximum(exp_var - noise_variance_, 0.)
covsk = np.dot(components_.T * exp_var_diff, components_)
covsk.flat[::len(covsk) + 1] += noise_variance_
print(covsk)