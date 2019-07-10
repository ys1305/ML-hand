import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

# 带正则化的LR
# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0,1,-1]])
    # print(data)
    return data[:,:2], data[:,-1]


#定义sigmoid函数
def sigmoid(z):
    return(1 / (1 + np.exp(-z)))

def fit(X_train, y_train, eta=0.01, n_iters=1e4):
    """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
    assert X_train.shape[0] == y_train.shape[0], \
        "the size of X_train must be equal to the size of y_train"


    def costfunc(theta, X_b, y,lam =1):
        # 计算损失函数
        y_hat = sigmoid(X_b.dot(theta))
        try:
            return -np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat)) / len(y) \
                     + (lam/(2.0*len(y)))*np.sum(np.square(theta[1:]))
                     # 不对截距做限制
        except:
            return float('inf')

    def dJ(theta, X_b, y,lam=1):
        # 损失函数求导
        y_hat = sigmoid(X_b.dot(theta))
        return X_b.T.dot(y_hat - y) / len(y) + (lam/len(y))*np.r_[[0],theta[1:]]

    def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):

        theta = initial_theta
        cur_iter = 0
        print('X_b.dot(theta)=',(X_b.dot(theta)).shape)
        print('(X_b.dot(theta) - y).shape=',(X_b.dot(theta) - y).shape)
        print('X_b.T.dot(X_b.dot(theta) - y).shape=',X_b.T.dot(X_b.dot(theta) - y).shape)


        while cur_iter < n_iters:
            gradient = dJ(theta, X_b, y)
            last_theta = theta
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

    X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
    proba = sigmoid(X_b.dot(theta))
    return np.array(proba >= 0.5, dtype='int')

def test():
    X, y = create_data()

    weight = fit(X,y)

    x_ponits = np.arange(4, 8)
    y_ = -(weight[1]*x_ponits + weight[0])/weight[2]


    print(weight)

    clf = LogisticRegression(
        # max_iter=200,
        C=1)
    clf.fit(X, y)
    print(clf.intercept_,clf.coef_)

    y_2 = -(clf.coef_[0][0]*x_ponits + clf.intercept_[0])/clf.coef_[0][1]

    theta2 = np.array([clf.intercept_[0],clf.coef_[0][0],clf.coef_[0][1]])

    plt.plot(x_ponits, y_,label='ys-lr')
    plt.plot(x_ponits, y_2,label='sklearn')
    plt.scatter(X[y==0,0], X[y==0,1])
    plt.scatter(X[y==1,0], X[y==1,1])
    plt.legend()
    plt.show()

def testsklearn():
    clf = LogisticRegression()
    X, y = create_data()
    clf.fit(X, y)
    print(clf.intercept_,clf.coef_)

if __name__ == '__main__':
    test()
    # testsklearn()

