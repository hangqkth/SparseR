import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
import scipy.sparse as sparse
import scipy.stats as stats
import matplotlib.pyplot as plt


# generate matrix A with full row rank
measure_num = 50
sparsity = 0.30


A = np.zeros((measure_num, 100))
for i in range(measure_num):
    for j in range(measure_num):
        A[i][j] = np.random.normal()
    np.random.shuffle(A[i])

# specify probability distribution
rvs = stats.norm(loc=0, scale=1).rvs  # standard Gaussian
# rvs = stats.laplace().rvs  # Laplace


def simulate_cvxpy(sparsity, A, eps=0.005):
    """Disciplined convex programming (DCP) to solve min or max problem"""
    # generate X from sparse function, and generate Y = A @ X
    S = sparse.random(100, 1, density=sparsity, data_rvs=rvs)
    X = S.toarray()
    Y = np.matmul(A, X)

    X_hat1 = cp.Variable((100, 1))
    X_hat2 = cp.Variable((100, 1))

    objective1 = cp.Minimize(cp.norm(X_hat1, 1))
    objective2 = cp.Minimize(cp.norm(X_hat2, 1))

    constraints1 = [A @ X_hat1 == Y]
    constraints2 = [cp.sum(cp.square(A @ X_hat2 - Y)) <= eps**2]

    prob1 = cp.Problem(objective1, constraints1)
    prob2 = cp.Problem(objective2, constraints2)


    prob1.solve()
    prob2.solve()

    print(cp.norm((A @ X_hat1) - Y, 2).value)
    print(cp.norm((A @ X_hat2) - Y, 2).value)

    # calculating ||X_hat-X||
    norm = np.linalg.norm(X_hat1.value - X, 2)

    plt.plot(X[:, 0])
    plt.plot(X_hat1.value)
    plt.ylabel("value of element in x or x_hat")
    plt.xlabel("sparse vector x and x_hat")
    plt.legend(["x", "x_hat"])
    plt.title("A@X_hat=Y, eps="+str(eps)+", sparsity="+str(sparsity)+", measurement="+str(measure_num))
    plt.show()

    plt.plot(X[:, 0])
    plt.plot(X_hat2.value)
    plt.ylabel("value of element in x or x_hat")
    plt.xlabel("sparse vector x and x_hat")
    plt.legend(["x", "x_hat"])
    plt.title("||A@X_hat-Y||2<=eps, eps="+str(eps)+", sparsity="+str(sparsity)+", measurement="+str(measure_num))
    plt.show()




def simulate_linprog(sparsity, A):
    # generate X and Y
    S = sparse.random(100, 1, density=sparsity, data_rvs=rvs)
    X = S.toarray()
    # print(X)
    Y = np.matmul(A, X)
    A_eq = A
    b_eq = Y
    c = np.ones((100, 1))
    bounds = [(None, None) for i in range(100)]
    print(A_eq.shape, b_eq.shape)
    res = linprog(c, A_ub=A_eq, b_ub=b_eq, bounds=bounds, method='highs')

    x_hat = np.expand_dims(res.x, axis=1)
    print(res.x)
    print(res.status)
    error_norm = np.linalg.norm(x_hat - X, 2)
    print("||X_hat-X||2 = ", error_norm)
    plt.plot(X)
    plt.plot(x_hat)
    plt.show()
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


simulate_cvxpy(sparsity=sparsity, A=A)
# e2 = simulate_linprog(sparsity=0.05, A=A)


