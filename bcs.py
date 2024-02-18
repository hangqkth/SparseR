import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.stats as stats
from omp_sp import omp_solution, sp_solution
from tqdm import tqdm


class BcsRvm:
    def __init__(self, num_nonzero, is_Cov=False, alpha=1.0, beta=1.0, Cov=None):
        self.alpha = alpha  # Precision of the prior on signal
        self.beta = beta    # Precision of the noise
        self.k = num_nonzero
        self.Cov = Cov
        self.is_Cov = is_Cov

    def n_argmin(self, arr, n):
        arr1 = arr.copy()
        n_idx_list = []
        for i in range(n):
            n_temp = np.argmin(np.abs(arr1))
            arr1[n_temp] = np.inf
            n_idx_list.append(n_temp)
        return n_idx_list

    def reshape_matrix(self, v, n):
        return np.reshape(v, (n, n)).T

    def fit(self, A, y, max_iter=1000):
        M, N = A.shape
        self.A = A  # Add a column of ones for the bias term
        self.y = y
        print(self.A.dtype)
        if M != y.shape[0]:
            print('matrix shape do not match')
            exit()

        self.alpha_vec = self.alpha * np.ones((N, 1))  # Initialize precision of weights

        mean = None
        for iter in range(max_iter):
            print(iter)
            # update posterior
            if self.is_Cov:
                # print((self.beta*self.Cov).shape)
                # print(np.dot(self.A.T, self.A).shape)
                Sigma_inv = np.diag(self.alpha_vec.ravel()) + self.beta * (self.A.T @ np.linalg.inv(self.Cov @ self.Cov.T) @ self.A)
                Sigma = np.linalg.inv(Sigma_inv)
                mean = self.beta * Sigma @ (self.A.T @ np.linalg.inv(self.Cov @ self.Cov.T) @ self.y)

            else:
                Sigma_inv = np.diag(self.alpha_vec.ravel()) + self.beta * self.A.T @ self.A
                Sigma = np.linalg.inv(Sigma_inv)
                mean = self.beta * np.dot(Sigma, np.dot(self.A.T, self.y))

            # update prior of x and noise
            gamma = 1 - self.alpha_vec * np.expand_dims(np.diag(Sigma), 1)

            for a in range(self.alpha_vec.shape[0]):
                value_temp = gamma[a, 0].copy() / np.square(mean[a, 0]).copy() if mean[a, 0] != 0 else 1e5
                value_temp = 1e10 if value_temp > 1e10 else value_temp  # in case the value too large and overflow
                self.alpha_vec[a, 0] = value_temp
            # self.alpha_vec = gamma / np.square(mean)
            # temp = np.linalg.norm((self.y - self.A @ mean), 2)**2
            temp = np.sum(np.square(self.y - self.A @ mean))
            # print(temp)
            self.beta = (M - np.sum(gamma)) / temp

            self.supports = self.n_argmin(self.alpha_vec, self.k)
            output = np.zeros(mean.shape)
            output[self.supports] = mean[self.supports]
            output = self.reshape_matrix(output, 50)

            plt.figure()
            plt.imshow(output)
            plt.colorbar()
            plt.show()

            # if self.is_Cov:
            #     b_a = np.kron(np.identity(50), output)
            #     self.Cov = np.linalg.inv(b_a @ b_a.T)

        self.supports = self.n_argmin(self.alpha_vec, self.k)
        output = np.zeros(mean.shape)
        output[self.supports] = mean[self.supports]

        return output


if __name__ == "__main__":
    # Example usage:
    m = 25 # y dim, number of measurement
    n = 50  # x dim, dimension of the sparse signal
    k = 15  # number of supports in x
    density = k / n  # sparsity

    rvs = stats.norm(loc=0, scale=1).rvs  # standard Gaussian
    A = np.zeros((m, n))  # fat matrix
    for i in range(m):
        for j in range(n):
            A[i][j] = np.random.normal()  # .normal() .laplace() or .poisson() or .uniform()
        np.random.shuffle(A[i])

    X = sparse.random(n, 1, density=density, data_rvs=rvs).toarray()
    Y = A @ X

    rvm = BcsRvm(k)
    x_hat_bcs = rvm.fit(A, Y)
    x_hat_omp = omp_solution(A, Y, k)
    x_hat_sp = sp_solution(A, Y, k)

    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.plot(range(len(x_hat_omp)), np.zeros(len(x_hat_omp)), c='black')

    plt.plot(np.nonzero(X)[0], X[np.nonzero(X)[0]], 'o', c='r')
    p1 = plt.vlines(range(len(X)), 0, X, colors='r')

    plt.plot(np.nonzero(x_hat_omp)[0], x_hat_omp[np.nonzero(x_hat_omp)], 'o', c='b')
    p2 = plt.vlines(range(len(x_hat_omp)), 0, x_hat_omp, colors='b')
    plt.legend(handles=[p1, p2], labels=['True X', 'Estimated X OMP'])
    plt.title("OMP, " + "k=" + str(k) + ", m=" + str(m) + ", n=" + str(n))

    plt.subplot(3, 1, 2)
    plt.plot(range(len(x_hat_sp)), np.zeros(len(x_hat_sp)), c='black')
    plt.plot(np.nonzero(X)[0], X[np.nonzero(X)[0]], 'o', c='r')
    p1 = plt.vlines(range(len(X)), 0, X, colors='r')
    #
    plt.plot(np.nonzero(x_hat_sp)[0], x_hat_sp[np.nonzero(x_hat_sp)], 'o', c='g')
    p3 = plt.vlines(range(len(x_hat_sp)), 0, x_hat_sp, colors='g')
    plt.legend(handles=[p1, p3], labels=['True X', 'Estimated X SP'])
    plt.title("SP, " + "k=" + str(k) + ", m=" + str(m) + ", n=" + str(n))

    plt.subplot(3, 1, 3)
    plt.plot(range(len(x_hat_bcs)), np.zeros(len(x_hat_bcs)), c='black')
    plt.plot(np.nonzero(X)[0], X[np.nonzero(X)[0]], 'o', c='r')
    p1 = plt.vlines(range(len(X)), 0, X, colors='r')
    #
    plt.plot(np.nonzero(x_hat_bcs)[0], x_hat_bcs[np.nonzero(x_hat_bcs)], 'o', c='black')
    p3 = plt.vlines(range(len(x_hat_bcs)), 0, x_hat_bcs, colors='black')
    plt.legend(handles=[p1, p3], labels=['True X', 'Estimated X BCS'])
    plt.title("BCS, " + "k=" + str(k) + ", m=" + str(m) + ", n=" + str(n))
    plt.show()


