import scipy.io as scio
import numpy as np
import os
import scipy.stats as stats
import cvxpy as cp
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from utils.compare_model import zeta_cutoff, compare_f1, get_roc


def omp_solution(A, Y, k):
    """
        Perform Orthonormal Matching Pursuit to approximate y using k dictionary elements from A.
        - A: The dictionary (matrix of shape (n, m)).
        - y: The signal to be approximated (vector of shape (n, 1)).
        - k: The desired sparsity level.
        - y = Ax + w or y = Ax
    """
    n, m = A.shape
    x_hat = np.zeros((m, 1))
    res = Y.copy()

    selected_columns = []
    A_selected = None

    for _ in range(k):
        h = np.abs(A.T @ res)
        selected_idx = np.argmax(h)
        while selected_idx in selected_columns:
            h[selected_idx] = 0
            selected_idx = np.argmax(np.abs(h))

        # Add the selected column to the support set.
        selected_columns.append(selected_idx)

        # update residual and A_ik
        A_selected = A[:, selected_columns]
        res = Y - A_selected @ np.linalg.pinv(A_selected) @ Y

    x_ik = np.linalg.pinv(A_selected) @ Y
    for i in range(len(selected_columns)):
        x_hat[selected_columns[i]] = x_ik[i]

    return x_hat


def nargmax(arr, n):
    arr1 = arr.copy()
    # print(arr1)
    n_idx_list = []
    for i in range(n):
        n_temp = np.argmax(np.abs(arr1))
        arr1[n_temp] = 0
        n_idx_list.append(n_temp)
    # print(n_idx_list)
    return n_idx_list


def sp_solution(A, Y, k):
    n, m = A.shape

    h = A.T @ Y
    idx_k = nargmax(h, k)

    A_ik = A.copy()[:, idx_k]

    res = Y - A_ik @ np.linalg.pinv(A_ik) @ Y

    iter = 0

    for _ in range(1000):
        iter += 1

        h = A.copy().T @ res

        idx_p = nargmax(h, k)
        # print(idx_p)

        # Add the selected column to the support set.
        idx_u = list(set(idx_k.copy() + idx_p))
        # print(idx_k.copy() + idx_p)

        A_iu = A.copy()[:, idx_u]
        # print(len(idx_u))
        x_iu = np.linalg.pinv(A_iu) @ Y

        x_hat_iu = np.zeros((m, 1))
        for i in range(len(idx_u)):
            x_hat_iu[idx_u[i]] = x_iu[i]

        idx_k = nargmax(x_hat_iu, k)

        A_ik = A.copy()[:, idx_k]
        res_new = Y - A_ik @ np.linalg.pinv(A_ik) @ Y

        if np.linalg.norm(res_new, 2) > np.linalg.norm(res):
            x_hat = np.zeros((m, 1))
            x_ik = np.linalg.pinv(A_ik) @ Y
            for j in range(len(idx_k)):
                x_hat[idx_k[j]] = x_ik[j]
            return x_hat
        else:
            res = res_new

    x_hat = np.zeros((m, 1))
    x_ik = np.linalg.pinv(A_ik) @ Y
    for j in range(len(idx_k)):
        x_hat[idx_k[j]] = x_ik[j]

    return x_hat


if __name__ == "__main__":
    # Basic setup y = Ax,
    n = 40
    m = 100
    k = 20
    density = k/m
    snr_omp = []
    snr_sp = []
    alpha_list = []

    # for n in range(98):
    #     x_norm_sum = []
    #     error_norm_sum_omp = []
    #     error_norm_sum_sp = []
    #
    #     for r in range(10):
    #         print(n)
    #         rvs = stats.norm(loc=0, scale=1).rvs  # standard Gaussian, or .laplace, .cosine, .uniform
    #         A = np.zeros((n+2, m))
    #
    #         for i in range(n+2):
    #             for j in range(m):
    #                 A[i][j] = np.random.normal()  # .normal() .laplace() or .poisson() or .uniform()
    #             np.random.shuffle(A[i])
    #
    #         X = sparse.random(m, 1, density=density, data_rvs=rvs).toarray()
    #         X = np.where(X <= 0, X, 1)
    #         X = np.where(X >= 0, X, -1)
    #         Y = A @ X
    #
    #         # Perform Orthonormal Matching Pursuit.
    #         x_hat_omp = omp_solution(A, Y, k)
    #         x_hat_sp = sp_solution(A, Y, k)
    #         # snr_omp.append(10*np.log10(np.linalg.norm(X, 2)/np.linalg.norm(X-x_hat_omp, 2)))
    #         # snr_sp.append(10 * np.log10(np.linalg.norm(X, 2) / np.linalg.norm(X-x_hat_sp, 2)))
    #         x_norm_sum.append(np.linalg.norm(X, 2))
    #         error_norm_sum_omp.append(np.linalg.norm(X - x_hat_omp, 2))
    #         error_norm_sum_sp.append(np.linalg.norm(X - x_hat_sp, 2))
    #     alpha = (n+1)/m
    #     alpha_list.append(alpha)
    #     snr_omp.append(10 * np.log10(sum(x_norm_sum) / sum(error_norm_sum_omp)))
    #     snr_sp.append(10 * np.log10(sum(x_norm_sum) / sum(error_norm_sum_sp)))
    #
    # print(len(alpha_list), len(snr_sp))
    # plt.subplot(2, 1, 1)
    # plt.plot(alpha_list, snr_omp)
    # plt.title('omp')
    # plt.subplot(2, 1, 2)
    # plt.plot(alpha_list, snr_sp)
    # plt.title('sp')
    # plt.show()
    # #
    #
    #
    #
    rvs = stats.norm(loc=0, scale=1).rvs  # standard Gaussian, or .laplace, .cosine, .uniform
    A = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            A[i][j] = np.random.normal()  # .normal() .laplace() or .poisson() or .uniform()
        np.random.shuffle(A[i])

    X = sparse.random(m, 1, density=density, data_rvs=rvs).toarray()
    # X = np.where(X <= 0, X, 1)
    # X = np.where(X >= 0, X, -1)

    Y = A @ X
    # Perform Orthonormal Matching Pursuit.
    x_hat_omp = omp_solution(A, Y, k)
    x_hat_sp = sp_solution(A, Y, k)
    #
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(range(len(x_hat_omp)), np.zeros(len(x_hat_omp)), c='black')

    plt.plot(np.nonzero(X)[0], X[np.nonzero(X)[0]], 'o', c='r')
    p1 = plt.vlines(range(len(X)), 0, X, colors='r')

    plt.plot(np.nonzero(x_hat_omp)[0], x_hat_omp[np.nonzero(x_hat_omp)], 'o', c='b')
    p2 = plt.vlines(range(len(x_hat_omp)), 0, x_hat_omp, colors='b')
    plt.legend(handles=[p1, p2], labels=['True X', 'Estimated X OMP'])
    plt.title("OMP, "+"k="+str(k))

    plt.subplot(2, 1, 2)
    plt.plot(range(len(x_hat_sp)), np.zeros(len(x_hat_sp)), c='black')
    plt.plot(np.nonzero(X)[0], X[np.nonzero(X)[0]], 'o', c='r')
    p1 = plt.vlines(range(len(X)), 0, X, colors='r')
    #
    plt.plot(np.nonzero(x_hat_sp)[0], x_hat_sp[np.nonzero(x_hat_sp)], 'o', c='g')
    p3 = plt.vlines(range(len(x_hat_sp)), 0, x_hat_sp, colors='g')
    plt.legend(handles=[p1, p3], labels=['True X', 'Estimated X SP'])
    plt.title("SP, "+"k="+str(k))
    plt.show()




