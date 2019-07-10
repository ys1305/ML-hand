# @Time    : 2019/1/8 13:54
# @Author  : Xu Huipeng
# @Blog    : https://brycexxx.github.io/

"""
双硬币模型

假设有两枚硬币A、B，以相同的概率随机选择一个硬币，进行如下的抛硬币实验：共做5次实验，每次实验独立的抛十次，
例如某次实验产生了H、T、T、T、H、H、T、H、T、H，H代表正面朝上。

假设实习生忘了记录每次试验选择的是 A 还是 B，我们无法观测实验数据中选择的硬币是哪个，数据如下：
硬币投掷结果观测序列
observations = np.array([[1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                         [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                         [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                         [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                         [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]])

问如何估计两个硬币正面出现的概率？

题目来源：http://www.hankcs.com/ml/em-algorithm-and-its-generalization.html
"""
import numpy as np
from scipy.stats import binom


class ExpectationMaximization:
    """
    简明 EM(期望最大化) 算法实现
    """

    def __init__(self, theta_a: float = 0.5, theta_b: float = 0.5, eps: float = 1e-3):
        self.eps = eps
        self.theta_a = theta_a
        self.theta_b = theta_b

    def fit(self, X: np.ndarray):
        # 初始化两枚硬币出现正面的概率
        n = X.shape[1]
        while True:
            counts = np.zeros((2, 2))
            for x in X:
                obverse_freq = x.sum()
                p_from_a = binom.pmf(obverse_freq, n, self.theta_a)
                p_from_b = binom.pmf(obverse_freq, n, self.theta_b)
                # 正规化
                p_from_a_normalized = p_from_a / (p_from_a + p_from_b)
                p_from_b_normalized = p_from_b / (p_from_a + p_from_b)
                counts[0, 0] += p_from_a_normalized * obverse_freq
                counts[0, 1] += p_from_a_normalized * (n - obverse_freq)
                counts[1, 0] += p_from_b_normalized * obverse_freq
                counts[1, 1] += p_from_b_normalized * (n - obverse_freq)
            # 更新 theta
            theta_a_old, theta_b_old = self.theta_a, self.theta_b
            self.theta_a = counts[0, 0] / counts[0, :].sum()
            self.theta_b = counts[1, 0] / counts[1, :].sum()
            if np.linalg.norm([self.theta_a - theta_a_old, self.theta_b - theta_b_old]) < self.eps:
                break
        return self


if __name__ == "__main__":
    x = np.array([[1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                  [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                  [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                  [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                  [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]])
    em = ExpectationMaximization(0.999999999, 0.000001)
    em.fit(x)
    print(em.theta_a, em.theta_b)
