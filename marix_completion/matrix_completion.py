import numpy as np
from scipy.sparse.linalg import svds


# R = np.array([[1, 1, 0, 0, 0],
#               [0, 0, 1, 0, 0],
#               [0, 0, 0, -1, 0],
#               [0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0]])
R = np.array([[0, 0, -1, 0, 0],
              [0, 0, 0, 1, 0],
              [1, 1, -1, 1, -1],
              [1, 0, 0, 0, -1],
              [0, 0, -1, 0, 0]])

# singular value decomposition

U, s, Vt = svds(A=R.astype(np.float64), k=4)
# s = np.flip(s)
print(s)
S = np.diag(s)
print(S)


R_hat = np.dot(np.dot(U, S), Vt)
print(R_hat)


R_hat = np.where(R_hat > 0, 1, -1)

print(R_hat)

