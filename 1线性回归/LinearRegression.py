import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# 形状非常重要，而且容易错误

def fit_normal(X_train, y_train):
    """根据训练数据集X_train, y_train训练Linear Regression模型"""
    assert X_train.shape[0] == y_train.shape[0], \
        "the size of X_train must be equal to the size of y_train"

    # np.vstack():在竖直方向上堆叠
    # np.hstack():在水平方向上平铺
    X_b = np.hstack([np.ones((len(X_train), 1)), X_train]) # 为了增加常数项
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

    intercept = theta[0]
    coef = theta[1:]

    return theta

def fit_bgd(X_train, y_train, eta=0.01, n_iters=1e5):
    """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
    assert X_train.shape[0] == y_train.shape[0], \
        "the size of X_train must be equal to the size of y_train"


    def costfunc(theta, X_b, y):
        # 计算损失函数
        try:
            return np.sum((y - X_b.dot(theta)) ** 2) / len(y)/2
        except:
            return float('inf')

    def dJ(theta, X_b, y):
        # 损失函数求导
        return X_b.T.dot(X_b.dot(theta) - y) / len(y)

    def gradient_descent(X_b, y, initial_theta, eta, n_iters=n_iters, epsilon=1e-8):

        theta = initial_theta
        cur_iter = 0
        print('X_b.dot(theta)=',(X_b.dot(theta)).shape)
        print('(X_b.dot(theta) - y).shape=',(X_b.dot(theta) - y).shape)
        print('X_b.T.dot(X_b.dot(theta) - y).shape=',X_b.T.dot(X_b.dot(theta) - y).shape)

        # y = np.array(data[:,1])时的维度
        # y_train.shape= (97,)
        # theta.shape= (2,)
        # X_b.dot(theta)= (97,)
        # (X_b.dot(theta) - y).shape= (97,)
        # X_b.T.dot(X_b.dot(theta) - y).shape= (2,)


        # y = np.c_[data[:,1]]时的维度
        # y_train.shape= (97, 1)
        # theta.shape= (2,)
        # X_b.dot(theta)= (97,)
        # (X_b.dot(theta) - y).shape= (97, 97)
        # X_b.T.dot(X_b.dot(theta) - y).shape= (2, 97)
        # ValueError: operands could not be broadcast together with shapes (2,) (2,97) 


        while cur_iter < n_iters:
            gradient = dJ(theta, X_b, y)
            # print((X_b.dot(theta)).shape)
            last_theta = theta
            # print(gradient.shape)
            theta = theta - eta * gradient
            if (abs(costfunc(theta, X_b, y) - costfunc(last_theta, X_b, y)) < epsilon):
                break

            cur_iter += 1

        return theta

    X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
    print('X_b.shape=',X_b.shape)
    print('y_train.shape=',y_train.shape)
    initial_theta = np.zeros(X_b.shape[1]) #初始化theta
    print('theta.shape=',initial_theta.shape)
    theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)

    intercept_ = theta[0]
    coef_ = theta[1:]

    return theta

def predict(X_predict,theta):
    """给定待预测数据集X_predict，返回表示X_predict的结果向量"""

    X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
    return X_b.dot(theta)

def test():
    data = np.loadtxt('linear_regression_data1.txt', delimiter=',')
    X = np.c_[data[:,0]]
    y = np.array(data[:,1])
    y1 = np.c_[data[:,1]]
    print(fit_normal(X,y))
    print(fit_bgd(X,y))

    regr = LinearRegression()
    regr.fit(X, y)
    print(regr.intercept_,regr.coef_)

def test0425():
    # 加载数据
    diabets = load_diabetes()
    data = diabets.data
    target = diabets.target

    # 打乱数据
    X, y = shuffle(data, target, random_state=13)

    # 划分训练集和测试集
    offset = int(X.shape[0] * 0.9)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    X=X_train
    y=y_train
    print(fit_normal(X,y))
    print(fit_bgd(X,y.reshape(len(y))))

    regr = LinearRegression()
    regr.fit(X, y)
    print(regr.intercept_,regr.coef_)


if __name__ == '__main__':
    test0425()

# ValueError: operands could not be broadcast together with shapes (2,) (2,97) 
