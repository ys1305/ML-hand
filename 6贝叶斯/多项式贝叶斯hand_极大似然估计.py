# coding:utf-8


class naiveBayes_MLE:
    def __init__(self, X, Y, N, n, K, x):
        self.X = X  # 训练数据的特征
        self.Y = Y  # 训练数据的类标记
        self.N = N  # 训练数据个数
        self.n = n  # 特征的个数
        self.K = K  # 类标记的个数
        self.x = x  # 待分类实例
 
    def prob(self):
        # 先验概率
        prior = {}
        # 条件概率
        conditional = {}
        # Y 训练数据的类标记 
        # set(Y):总共几类,1和-1
        for c in set(self.Y):
            prior[c] = 0
            conditional[c] = {}
            for j in range(self.n):
                for a in set(self.X[j]):
                    conditional[c][a] = 0
        print(conditional)
        #  set(self.X[1])为 1,2,3 ; set(self.X[2) 为S,L,M
        # {1: {1: 0, 2: 0, 3: 0, 'S': 0, 'L': 0, 'M': 0},
        # -1: {1: 0, 2: 0, 3: 0, 'S': 0, 'L': 0, 'M': 0}}
        
        # 计算先验概率和条件概率
        # N为样本数量
        # prior:{1:0,-1:0} 记录每个类别的数量
        for i in range(self.N):
            # Y[i]=1或-1
            prior[self.Y[i]] += 1
            # n为特征的个数
            for j in range(self.n):
                # X[1][i]的取值为1,2,3
                # X[2][i]的取值为S,L,M
                conditional[self.Y[i]][self.X[j][i]] += 1
        # 除以样本总数N之后，得到真正意义上的先验概率和条件概率
        for c in set(self.Y):
            for j in range(self.n):
                for a in set(self.X[j]):
                    conditional[c][a] /= prior[c]
 
            prior[c] /= self.N
 
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


# +

X = [[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
     ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L']]
Y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
N = len(Y)
n = len(X)
K = len(set(Y))
x = [2, 'S']

nb = naiveBayes_MLE(X, Y, N, n, K, x)
posterior, argmax = nb.classifier()
print("每个类别的后验概率:", posterior)
print("x=", x, "的类标记y为", argmax)
# -

nb.prob()


