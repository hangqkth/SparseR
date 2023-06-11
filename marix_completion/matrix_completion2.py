import numpy as np
import cvxpy as cp


# A = np.zeros((5, 5))
# A[0, 1] = 1
# A[0, 3] = 1
# A[1, 2] = -1
# A[1, 3] = 1
# A[2, 2] = -1
# A[2, 4] = 1
# A[3, 0] = -1
# A[3, 3] = 1
A = np.array([[0, 0, -1, 0, 0],
              [0, 0, 0, 1, 0],
              [1, 1, -1, 1, -1],
              [1, 0, 0, 0, -1],
              [0, 0, -1, 0, 0]])

# minimization of nuclear norm to minimize rank
X = cp.Variable((5, 5))
objective = cp.Minimize(cp.norm(X, "nuc"))
constraints = [X[i, j] == A[i, j] for i in range(5) for j in range(5) if A[i, j] != 0]  # unknown entries set to 0
problem = cp.Problem(objective, constraints)


result = problem.solve()

if problem.status == 'optimal':
    print("Optimal value: ", result)
    print("Recovered matrix:\n", np.round(X.value))
else:
    print("Problem status: ", problem.status)

X.value = np.where(X.value > 0, 1, -1)
print("Recovered matrix:\n", X.value)
