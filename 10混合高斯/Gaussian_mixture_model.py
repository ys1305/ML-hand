# @Time    : 2019/1/9 14:07
# @Author  : Xu Huipeng
# @Blog    : https://brycexxx.github.io/
import numpy as np
from typing import Any


class GaussianMixture:
    """
    高斯混合模型
    """

    def __init__(self, n_components: int = 1, eps: float = 1e-3,
                 max_iter: int = 200, random_state: Any = None):
        self.n_components = n_components
        self.eps = eps
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = None
        self.miu = None
        self.sigma_square = None

    def fit(self, X: np.ndarray):
        m = X.shape[0]
        rs = np.random.RandomState(self.random_state)
        # 初始化模型参数
        alpha = rs.random_sample((1, self.n_components))
        self.alpha = alpha / alpha.sum()
        self.miu = rs.random_sample((1, self.n_components))
        self.sigma_square = rs.random_sample((1, self.n_components))
        for _ in range(self.max_iter):
            # E 步
            gamma = self.alpha * (1.0 / np.sqrt(2 * np.pi * self.sigma_square) *
                                  (np.exp(-(X - self.miu) ** 2 / (2 * self.sigma_square)) + 1e-9))
            gamma = gamma / gamma.sum(axis=1, keepdims=True)
            # M 步
            miu_old = self.miu.copy()
            sigma_square_old = self.sigma_square.copy()
            alpha_old = self.alpha.copy()
            self.miu = (gamma * X).sum(axis=0) / gamma.sum(axis=0, keepdims=True)
            self.sigma_square = (gamma * (X - self.miu) ** 2).sum(axis=0, keepdims=True) \
                                / gamma.sum(axis=0, keepdims=True)
            self.alpha = gamma.sum(axis=0, keepdims=True) / m
            delta_alpha = self.alpha - alpha_old
            delta_miu = self.miu - miu_old
            delta_sigma_square = self.sigma_square - sigma_square_old
            if np.linalg.norm(delta_miu) < self.eps and np.linalg.norm(delta_sigma_square) < self.eps \
                    and np.linalg.norm(delta_alpha) < self.eps:
                break
        return self


if __name__ == "__main__":
    def generate_data(length, alpha0, alpha1, miu0, miu1, sigma0, sigma1):
        data = np.zeros((length, 1))
        data0 = np.random.normal(miu0, sigma0, int(alpha0 * length))
        data1 = np.random.normal(miu1, sigma1, int(alpha1 * length))
        data[:int(alpha0 * length), 0] = data0[:]
        data[int(alpha0 * length):, 0] = data1[:]
        np.random.shuffle(data)
        return data


    data = generate_data(1000, 0.1, 0.9, 12, 11, 0.2, 6)
    # 初始化观测数据
    data=np.array([-67, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60, 75]).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, eps=1e-5, max_iter=1000)
    gmm.fit(data)
    print(gmm.alpha)
    print(gmm.miu)
    print(np.sqrt(gmm.sigma_square))
