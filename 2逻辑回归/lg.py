import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification
from sklearn.linear_model import LogisticRegression

def initialize_params(dims):
    w = np.zeros((dims, 1))
    b = 0
    return w, b

def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z

def logistic(X, y, w, b):
    num_train = X.shape[0]
    y_hat = sigmoid(np.dot(X, w) + b)
    loss = -1 / num_train * np.sum(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))
    cost = -1 / num_train * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    dw = np.dot(X.T, (y_hat - y)) / num_train
    db = np.sum(y_hat - y) / num_train
    return y_hat, cost, dw, db

def linear_train(X, y, learning_rate, epochs):
    # 参数初始化
    w, b = initialize_params(X.shape[1])

    loss_list = []
    for i in range(epochs):
        # 计算当前的预测值、损失和梯度
        y_hat, loss, dw, db = logistic(X, y, w, b)
        loss_list.append(loss)

        # 基于梯度下降的参数更新
        w += -learning_rate * dw
        b += -learning_rate * db

        # 打印迭代次数和损失
        if i % 10000 == 0:
            print("epoch %d loss %f" % (i, loss))

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
    y_pred = sigmoid(np.dot(X, w) + b)
    return y_pred


if __name__ == "__main__":
    # 生成数据
    X, labels = make_classification(n_samples=100,
                                    n_features=2,
                                    n_informative=2,
                                    n_redundant=0,
                                    random_state=1,
                                    n_clusters_per_class=2)
    print(X.shape)
    print(labels.shape)

    X = np.array([[3, 3, 3],
                        [4, 3, 2],
                        [2, 1, 2],
                        [1, 1, 1],
                        [-1, 0, 1],
                        [2, -2, 1]])
    labels = np.array([1, 1, 1, 0, 0, 0])

    print(X.shape)
    print(labels.shape)


    # 生成伪随机数
    rng = np.random.RandomState(2)
    # X += 2 * rng.uniform(size=X.shape)

    # 划分训练集和测试集
    offset = int(X.shape[0] * 0.9)
    X_train, y_train = X[:offset], labels[:offset]
    X_test, y_test = X[offset:], labels[offset:]
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    # 一般情况y的shape为[samples,]
    print('X_train=', X_train.shape)
    print('y_train=', y_train.shape)
    print('X_test=', X_test.shape)
    print('y_test=', y_test.shape)

    # 训练
    loss_list, loss, params, grads = linear_train(X_train, y_train, 0.01, 100000)
    print(params)

    clf = LogisticRegression(max_iter=20000)
    clf.fit(X_train, y_train)
    print(clf.intercept_,clf.coef_)

    # 预测
    y_pred = predict(X_train, params)
    print(y_pred[:10])