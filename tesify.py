import numpy as np
import cvxpy as cp

# 生成矩阵A
A = np.zeros((50, 100))
for i in range(50):
    for j in range(50):
        A[i][j] = np.random.normal()
    np.random.shuffle(A[i])

# 生成向量X
X = np.random.normal(size=(100, 1))

# 计算Y
Y = np.matmul(A, X)

# 定义优化问题
X_hat = cp.Variable((100, 1))
objective = cp.Minimize(cp.norm(X_hat, 1))
constraints = [A @ X_hat == Y]
prob = cp.Problem(objective, constraints)

# 解决优化问题
prob.solve()

# 计算||X_hat-X||2
norm = np.linalg.norm(X_hat.value - X, 2)
print("||X_hat-X||2 = ", norm)
