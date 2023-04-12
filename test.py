import numpy as np
from scipy.optimize import linprog

# Define the problem data
A = np.array([[1, 2, 3, 9], [3, 4, 5, 2], [5, 6, 7, 15]])
b = np.array([16, 8, 1])
c = 2

# Define the objective coefficients
c_obj = np.ones(A.shape[1])

# Define the constraint coefficients
A_cons = np.concatenate((A, -A), axis=0)
b_cons = np.concatenate((b+c, -b+c), axis=0)
# print(A_cons.shape, b_cons.shape, c_obj.shape)

# bounds = [(None, None, None) for i in range(2*A.shape[0])]

result = linprog(c=c_obj, A_ub=A_cons, b_ub=b_cons, method='interior-point')
# Print the optimal value and solution
# print(result.x)




