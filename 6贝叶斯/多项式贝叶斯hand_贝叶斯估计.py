# coding:utf-8
"""
@author:hanmy
@file:nb_be.py
@time:2019/04/29
"""
class naiveBayes_BE:
    def __init__(self, X, Y, N, n, K, x, lamb):
        self.X = X  # 训练数据的特征
        self.Y = Y  # 训练数据的类标记
        self.N = N  # 训练数据个数
        self.n = n  # 特征的个数
        self.K = K  # 类标记的个数
        self.x = x  # 待分类实例
        self.lamb = lamb  # 贝叶斯估计的lambda
 
    def prob(self):
        # 先验概率
        prior = {}
        # 条件概率
        conditional = {}
        for c in set(self.Y):
            prior[c] = 0
            conditional[c] = {}
            for j in range(self.n):
                for a in set(self.X[j]):
                    conditional[c][a] = 0
        # 每个特征有多少个不同的特征值
        S = [0]*self.n
        for j in range(self.n):
            for _ in set(self.X[j]):
                S[j] += 1
 
        # 计算先验概率和条件概率
        for i in range(self.N):
            prior[self.Y[i]] += 1
            for j in range(self.n):
                conditional[self.Y[i]][self.X[j][i]] += 1
 
        for c in set(self.Y):
            for j in range(self.n):
                for a in set(self.X[j]):
                    conditional[c][a] = (conditional[c][a] + self.lamb) / (prior[c] + S[j]*self.lamb)
 
            prior[c] = (prior[c] + self.lamb) / (self.N + self.K*self.lamb)
 
        return prior, conditional
 
    # 确定实例x的类
    def classifier(self):
        prior, conditional = self.prob()
        # 计算各类别的后验概率
        posterior = {}
        for c in set(self.Y):
            cond = 1
            for j in range(self.n):
                cond *= conditional[c][self.x[j]]
            posterior[c] = prior[c] * cond
 
        # 取最大后验概率的类别max(dict, key=dict.get)
        argmax = max(posterior, key=posterior.get)
 
        return posterior, argmax
 
 
if __name__ == "__main__":
    X = [[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
         ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L']]
    Y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    N = len(Y)
    n = len(X)
    K = len(set(Y))
    x = [2, 'S']
    lamb = 0.2
 
    nb = naiveBayes_BE(X, Y, N, n, K, x, lamb)
    posterior, argmax = nb.classifier()
    print("每个类别的后验概率:", posterior)
    print("x=", x, "的类标记y为", argmax)
