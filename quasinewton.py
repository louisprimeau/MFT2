import numpy as np
from numpy import inf, zeros
from numpy.linalg import solve, inv
from numpy import concatenate as cat

def quasi_newton(func, x0, q=5, tol=1e-7, history=None):

    f0 = func(x0)
    n = f0.shape[0]
    df = inf

    alpha = 1e-3
    U = zeros((n, q)); 
    V = zeros((n, q)); 

    xn = x0; fn = f0
    
    # Build up U and V
    for i in range(0, q):
        xn_1 = xn - alpha * func(xn)
        fn = func(xn)
        U[:, i] = (fn - xn)
        V[:, i] = (func(fn) - fn)
        xn = xn_1

    # Run the full algorithm
    for i in range(100):
        print("iteration", i)
        fn = func(xn)
        xn_1 = fn - V @ (solve(U.T @ U - U.T @ V, U.T) @ (xn - fn))
        history.append(xn_1.reshape(1, -1))
        fn_1 = func(xn_1)
        U = cat((U[:, 1:],(fn_1 - xn_1).reshape(-1, 1)), axis=1)
        V = cat((V[:, 1:], (func(fn_1) - fn_1).reshape(-1, 1)), axis=1)
        dx = np.max(np.abs(xn_1 - xn))
        xn = xn_1
        print(dx)
        if dx < tol:
            print("Finished at iteration {}".format(i))
            break        

    return xn
