import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from scipy import io as spio
from scipy import misc      # 图片操作
import numbers
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 解决windows环境下画图汉字乱码问题

def distance(point1, point2):
    # 欧氏距离
    return np.sqrt(np.sum(np.square(point1 - point2), axis=1))
    # return np.sqrt(np.sum(np.power(point1 - point2,2)))


def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

random_state = check_random_state(None)
# kmeans++的初始化方式，加速聚类速度
# 第一个点是随机选择出来的
def k_means_plus_plus(dataset,k):
    n_samples, n_features = dataset.shape
    centers = np.empty((k, n_features))


    # n_local_trials是每次选择候选点个数
    n_local_trials = None
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(k))

    # 第一个随机点
    center_id = random_state.randint(n_samples)
    centers[0] = dataset[center_id]

    # closest_dist_sq是每个样本到所有中心点最近距离
    # 假设现在有3个中心点，closest_dist_sq = 
    # [min(样本1到3个中心距离),min(样本2到3个中心距离),...min(样本n到3个中心距离)]
    closest_dist_sq = distance(centers[0, np.newaxis], dataset)
    # newaxis可以给原数组增加一个维度

    # current_pot所有最短距离的和
    current_pot = closest_dist_sq.sum()

    for c in range(1, k):
        # 选出n_local_trials随机址，并映射到current_pot的长度
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        # 选择出来的候选节点是按照概率选择出来的
        # 然后再根据所有样本到候选节点的距离选择出来距离最小的节点

        # np.cumsum([1,2,3,4]) = [1, 3, 6, 10]，就是累加当前索引前面的值
        # np.searchsorted搜索随机出的rand_vals落在np.cumsum(closest_dist_sq)中的位置。
        # candidate_ids候选节点的索引
        candidate_ids = np.searchsorted(np.cumsum(closest_dist_sq), rand_vals)
        print(candidate_ids)

        # best_candidate最好的候选节点
        # best_pot最好的候选节点计算出的距离和
        # best_dist_sq最好的候选节点计算出的距离列表
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # 计算每个样本到候选节点的欧式距离
            distance_to_candidate = distance(dataset[candidate_ids[trial], np.newaxis], dataset)

            # 计算每个候选节点的距离序列new_dist_sq， 距离总和new_pot

            #   closest_dist_sq 每个样本，到所有已知的中心点的距离
            #   new_dist_sq 每个样本，到所有中心点（已知的中心点+当前的候选点）最近距离
            # 如果中心点变成了两个，那么样本到中心点的最近距离就可能会发生变化
            new_dist_sq = np.minimum(closest_dist_sq, distance_to_candidate)
            new_pot = new_dist_sq.sum()

            # 选择最小的new_pot
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        centers[c] = dataset[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers


##################### 相对简单版--理解
# 但是可能会出现初值不好的情况
# [[3.  0. ]
#  [0.  0. ]
#  [3.1 3.1]]

def distance1(point1, point2):
    # 欧氏距离
    return np.sqrt(np.sum(np.power(point1 - point2,2)))

#对一个样本找到与该样本距离最近的聚类中心
def nearest(point, cluster_centers):
    min_dist = np.inf
    m = np.shape(cluster_centers)[0]  # 当前已经初始化的聚类中心的个数
    for i in range(m):
        # 计算point与每个聚类中心之间的距离
        d = distance1(point, cluster_centers[i, ])
        # 选择最短距离
        if min_dist > d:
            min_dist = d
    return min_dist

#选择尽可能相距较远的类中心
def get_centroids(dataset, k):
    m, n = np.shape(dataset)
    cluster_centers = np.zeros((k , n))
    index = np.random.randint(0, m)
    # index = random_state.randint(0,m)
    # 返回一个随机整型数，范围从低（包括）到高（不包括）
    # print(index)
    cluster_centers[0] = dataset[index]
    # 2、初始化一个距离的序列
    d = [0.0 for _ in range(m)]
    for i in range(1, k):
        sum_all = 0
        for j in range(m):
            # 3、对每一个样本找到最近的聚类中心点
            d[j] = nearest(dataset[j], cluster_centers[0:i])
            # 4、将所有的最短距离相加
            sum_all += d[j]
        # 5、取得sum_all之间的随机值
        # print(d)
        sum_all *= np.random.rand()

        # np.searchsorted搜索随机出的sum_all落在np.cumsum(d)中的位置。等价于下面6的代码
        candidate_ids = np.searchsorted(np.cumsum(d), sum_all)
        cluster_centers[i] = dataset[candidate_ids]


        # ##6、获得距离最远的样本点作为聚类中心点
        # for j, di in enumerate(d):
        #     sum_all=sum_all - di
        #     if sum_all > 0:
        #         continue
        #     cluster_centers[i] = dataset[j]
        #     break
    return cluster_centers



data = np.array([[0.,0.],
	[0.1,0.1],[0.2,0.2],[3.0,0.0],[3.1,3.1],[3.2,3.2],[9.0,9.0],[9.1,9.1],[9.2,9.2]
	])

print(data)

print(k_means_plus_plus(data,3))
print(get_centroids(data,3))