import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
import scipy.sparse as sparse
import scipy.stats as stats

# 生成矩阵A,行满秩
A = np.zeros((50, 100))
for i in range(50):
    for j in range(50):
        A[i][j] = np.random.normal()
    np.random.shuffle(A[i])

# 生成向量X
# specify probability distribution
rvs = stats.norm(loc=3, scale=1).rvs
# create sparse random matrix with specific probability distribution/random numbers.
S = sparse.random(100, 1, density=0.15, data_rvs=rvs)
# Convert the sparse matrix to a full matrix
X = S.toarray()
print(X.shape)

# 计算Y
Y = np.matmul(A, X)
print(Y.shape)


# 定义优化问题
def simulate_cvxpy(X, Y, A):
    X_hat = cp.Variable((100, 1))
    objective = cp.Minimize(cp.norm(X_hat, 1))
    constraints = [A @ X_hat == Y]
    prob = cp.Problem(objective, constraints)

    # 解决优化问题
    prob.solve()

    # 计算||X_hat-X||2
    print(X_hat.value)
    norm = np.linalg.norm(X_hat.value - X, 2)
    print("||X_hat-X||2 = ", norm)


def simulate_linprog(X, Y, A):
    A_eq = A
    b_eq = Y
    c = np.ones((100, ))
    bounds = [(0, None) for i in range(100)]
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    x_hat = np.expand_dims(res.x, axis=1)
    error_norm = np.linalg.norm(x_hat - X, 2)
    print("||X_hat-X||2 = ", error_norm)


simulate_cvxpy(X, Y, A)
simulate_linprog(X, Y, A)


