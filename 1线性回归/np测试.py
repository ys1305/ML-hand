import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.c_[a,b]

print(np.r_[a,b])
print(c)
print(np.c_[c,a])
print(np.c_[c,a].size)

# np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，
# 类似于pandas中的merge()。
# [1 2 3 4 5 6]
# [[1 4]
#  [2 5]
#  [3 6]]
# [[1 4 1]
#  [2 5 2]
#  [3 6 3]]
# 9