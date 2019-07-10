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

def fit(X_train, y_train,lam, eta=0.01, n_iters=1e4):
    """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
    assert X_train.shape[0] == y_train.shape[0], \
        "the size of X_train must be equal to the size of y_train"


    def costfunc(theta, X_b, y,lam =1):
        # 计算损失函数
        y_hat = sigmoid(X_b.dot(theta))
        try:
            return -np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat)) / len(y) \
                     + (lam/(2.0*len(y)))*np.sum(np.square(theta[1:]))
        except:
            return float('inf')

    def dJ(theta, X_b, y,lam):
        # 损失函数求导
        y_hat = sigmoid(X_b.dot(theta))
        return X_b.T.dot(y_hat - y) / len(y) + (lam/len(y))*np.r_[[0],theta[1:]]

    def gradient_descent(X_b, y, initial_theta, lam,eta, n_iters=1e4, epsilon=1e-8):

        theta = initial_theta
        cur_iter = 0

        while cur_iter < n_iters:
            gradient = dJ(theta, X_b, y,lam)
            last_theta = theta
            theta = theta - eta * gradient
            if (abs(costfunc(theta, X_b, y,lam) - costfunc(last_theta, X_b, y,lam)) < epsilon):
                break

            cur_iter += 1

        return theta

    X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
    # X_b = np.hstack([np.zeros((len(X_train), 1)), X_train])
    initial_theta = np.zeros(X_b.shape[1]) #初始化theta
    theta = gradient_descent(X_b, y_train,initial_theta,lam ,eta, n_iters)

    intercept_ = theta[0]
    coef_ = theta[1:]

    return theta

def predict(X_predict,theta):

    X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
    # X_b = np.hstack([np.zeros((len(X_predict), 1)), X_predict])
    proba = sigmoid(X_b.dot(theta))
    return np.array(proba >= 0.6, dtype='int')

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

# if __name__ == '__main__':
#     test()
#     # testsklearn()
X = np.random.normal(0, 1, size=(200, 2))
y = np.array((X[:,0]**2+X[:,1])<1.5, dtype='int')


# data2 = np.loadtxt('data2.txt', delimiter=',')
# y = np.array(data2[:,2])
# X = np.array(data2[:,0:2])


poly = PolynomialFeatures(6)
XX = poly.fit_transform(X)

fig, axes = plt.subplots(1,3, sharey = True, figsize=(17,5))

# 决策边界，咱们分别来看看正则化系数lambda太大太小分别会出现什么情况
# Lambda = 0 : 就是没有正则化，这样的话，就过拟合咯
# Lambda = 1 : 这才是正确的打开方式
# Lambda = 100 : 卧槽，正则化项太激进，导致基本就没拟合出决策边界

for i, C in enumerate([0.0, 10.0, 1000.0]):
    # 最优化 costFunctionReg
    weight = fit(XX,y,lam=C)
    
    # 准确率
    accuracy = 100.0*sum(predict(XX,weight) == y.ravel())/y.size    

    # 对X,y的散列绘图

    # plt.scatter(X[y==0,0], X[y==0,1])
    # plt.scatter(X[y==1,0], X[y==1,1])
    axes.flatten()[i].scatter(X[y==0,0], X[y==0,1])
    axes.flatten()[i].scatter(X[y==1,0], X[y==1,1])
    # 画出决策边界
    x1_min, x1_max = X[:,0].min(), X[:,0].max(),
    x2_min, x2_max = X[:,1].min(), X[:,1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(weight[1:])+1*weight[0])
    h = h.reshape(xx1.shape)
    axes.flatten()[i].contour(xx1, xx2, h, [0.5], linewidths=1, colors='g');       
    axes.flatten()[i].set_title('Train accuracy {}% with Lambda = {}'.format(np.round(accuracy, decimals=2), C))
plt.show()