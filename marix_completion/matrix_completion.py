import numpy as np
from scipy.sparse.linalg import svds

# 创建一个5*5的矩阵，其中只有第2列的第1和3个元素是已知的评分，其余为缺失值
R = np.array([[0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]])

# 对R进行SVD分解，取前两个奇异值和相应的左右奇异向量
U, s, Vt = svds(R.astype(float), k=2)
S = np.diag(s)

# 低秩矩阵补全
R_hat = np.dot(np.dot(U, S), Vt)

# 将矩阵中大于0的元素设为1，小于等于0的元素设为-1
R_hat = np.where(R_hat > 0, 1, -1)

print(R_hat)

