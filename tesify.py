import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
import scipy.sparse as sparse
import scipy.stats as stats
import matplotlib.pyplot as plt


# generate matrix A with full row rank
measurement_num = 70
A = np.zeros((measurement_num, 100))
for i in range(measurement_num):
    for j in range(measurement_num):
        A[i][j] = np.random.normal()
    np.random.shuffle(A[i])

# specify probability distribution
rvs = stats.norm(loc=0, scale=1).rvs
# create sparse random matrix with specific probability distribution/random numbers.


# 定义优化问题
def simulate_cvxpy(sparsity, A):
    # generate X and Y
    S = sparse.random(100, 1, density=sparsity, data_rvs=rvs)
    X = S.toarray()
    X_hat = cp.Variable((100, 1))
    Y = np.matmul(A, X)
    objective = cp.Minimize(cp.norm(X_hat, 1))
    constraints = [A @ X_hat == Y]
    prob = cp.Problem(objective, constraints)

    # solve the optimization problem
    prob.solve()

    # calculating ||X_hat-X||2
    # print(X_hat.value)
    norm = np.linalg.norm(X_hat.value - X, 2)
    print(X.shape)
    print(X_hat.shape)
    x_hat = list(X_hat.value[:, ])
    x_hat = [float(i) for i in x_hat]

    plt.plot(X[:, 0])
    plt.plot(X_hat.value)
    plt.ylabel("value of element in x or x_hat")
    plt.xlabel("sparse vector x and x_hat")
    plt.legend(["x", "x_hat"])
    plt.show()

    # print("||X_hat-X||2 = ", norm)
    return norm, X


def simulate_linprog(sparsity, A):
    # generate X and Y
    S = sparse.random(100, 1, density=sparsity, data_rvs=rvs)
    X = S.toarray()
    # print(X)
    Y = np.matmul(A, X)
    A_eq = A
    b_eq = Y
    c = np.ones((100, ))
    bounds = [(None, None) for i in range(100)]
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    x_hat = np.expand_dims(res.x, axis=1)
    error_norm = np.linalg.norm(x_hat - X, 2)
    print("||X_hat-X||2 = ", error_norm)
    # plt.plot(list(range(100)), X)
    # plt.plot(range(100), x_hat)
    # plt.show()
    return x_hat


def do_plot(s_min, s_max, A):
    e_list1, e_list2 = [], []
    for i in range(10):
        e_list1.append([simulate_cvxpy(s, A) for s in np.arange(s_min, s_max, 0.05)])
        e_list2.append([simulate_linprog(s, A) for s in np.arange(s_min, s_max, 0.05)])
    ea1, ea2 = np.average(np.array(e_list1), axis=0), np.average(np.array(e_list2), axis=0)
    # s_x = np.arange(s_min, s_max, 0.05)
    # plt.plot(s_x, ea1)
    # plt.plot(s_x, ea2)
    # plt.legend(['cvxpy', 'linprog'])
    # plt.xlabel("sparsity")
    # plt.ylabel("||x_hat-x||2")
    # plt.show()


e1 = simulate_cvxpy(sparsity=0.5, A=A)
# e2, x2 = simulate_linprog(sparsity=0.05, A=A)


