import numpy as np
import cvxpy as cp

# 定义一个稀疏矩阵，其中只有一小部分元素是已知的
A = np.zeros((5, 5))
A[0, 1] = 1
A[1, 2] = 1

# 定义核范数优化问题
X = cp.Variable((5, 5))
objective = cp.Minimize(cp.norm(X, "nuc"))
constraints = [X[i, j] == A[i, j] for i in range(5) for j in range(5) if A[i, j] != 0]
problem = cp.Problem(objective, constraints)

# 解决问题并输出结果
result = problem.solve()

if problem.status == 'optimal':
    print("Optimal value: ", result)
    print("Recovered matrix:\n", X.value)
else:
    print("Problem status: ", problem.status)

X.value = np.where(X.value > 0, 1, -1)
print("Recovered matrix:\n", X.value)
