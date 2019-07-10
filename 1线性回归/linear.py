import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def initialize_params(dims):
    w = np.zeros((dims, 1))
    b = 0
    return w, b


def linear_loss(X, y, w, b):
    num_train = X.shape[0]
    # 模型公式
    y_hat = np.dot(X, w) + b
    # 损失函数
    loss = np.sum((y_hat - y) ** 2) / num_train
    # 参数偏导
    dw = np.dot(X.T, (y_hat - y)) / num_train
    db = np.sum(y_hat - y) / num_train
    return y_hat, loss, dw, db


def linear_train(X, y, learning_rate, epochs):
    # 参数初始化
    w, b = initialize_params(X.shape[1])

    loss_list = []
    for i in range(1, epochs):
        # 计算当前预测值、损失和梯度
        y_hat, loss, dw, db = linear_loss(X, y, w, b)
        loss_list.append(loss)

        # 基于梯度下降的参数更新
        w += -learning_rate * dw
        b += -learning_rate * db

        # 打印迭代次数和损失
        if i % 10000 == 0:
            print('epoch %d loss %f' % (i, loss))

        # 保存参数
        params = {
            'w': w,
            'b': b
        }

        # 保存梯度
        grads = {
            'dw': dw,
            'db': db
        }
    return loss_list, loss, params, grads


def predict(X, params):
    w = params['w']
    b = params['b']
    y_pred = np.dot(X, w) + b
    return y_pred


if __name__ == "__main__":
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

    # 训练
    loss_list, loss, params, grads = linear_train(X_train, y_train, 0.01, 100000)
    print(params)

    regr = LinearRegression()
    regr.fit(X, y)
    print(regr.intercept_,regr.coef_)

    # 预测
    y_pred = predict(X_test, params)
    print(y_pred[:5])

    # 画图
    f = X_test.dot(params['w']) + params['b']
    plt.scatter(range(X_test.shape[0]), y_test)
    plt.plot(f, color='darkorange')
    plt.xlabel('x')
    plt.xlabel('y')
    plt.show()

    plt.plot(loss_list, color='blue')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()