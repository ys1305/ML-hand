import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import axes3d


data = np.loadtxt('linear_regression_data1.txt', delimiter=',')
X = np.c_[data[:,0]]
X1 = np.c_[np.ones(data.shape[0]),data[:,0]]
y1 = np.c_[data[:,1]]
y2 = np.array(data[:,1])
print('X1.shape=',X1.shape)
print('y1.shape=',y1.shape)
print('len(y1)=',len(y1))
print('y2.shape=',y2.shape)

initial_theta = np.zeros((X1.shape[1],1)) 
# 计算损失函数
def computeCost(X, y, theta=initial_theta):
    m = y.size
    J = 0
    
    h = X.dot(theta)
    
    J = 1.0/(2*m)*(np.sum(np.square(h-y)))
    
    return J

# 梯度下降
def gradientDescent(X, y, theta=initial_theta, alpha=0.01, num_iters=5000):
    m = y.size
    J_history = np.zeros(num_iters)


    theta1 = np.zeros(X.shape[1])
    print('theta1.shape=',theta1.shape)

    print('theta.shape=',theta.shape)
    for iter in np.arange(num_iters):
        h = X.dot(theta)
        theta = theta - alpha*(1.0/m)*(X.T.dot(h-y))
        J_history[iter] = computeCost(X, y, theta)
    return(theta, J_history)


theta , Cost_J = gradientDescent(X1, y1)
print('theta: ',theta.ravel())


# y1.shape= (97, 1)
# y2.shape= (97,)
# theta1.shape= (2,)
# theta.shape= (2, 1)