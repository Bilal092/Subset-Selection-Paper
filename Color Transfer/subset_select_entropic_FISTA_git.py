#%%
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
np.errstate(divide='ignore')
#%%
'''
Entropic regularized subset selection.
Written By: bilal Riaz (bilalria@udel.edu)

Disclaimer: The entropic regularization of linear programs is inspired from the 
            success of interior point optimization methods 1990's. This success comes
            with cost that these methods are inherently unstable because of logarithmic 
            barriers. If the entropic regularization-based optimization process returns NaN
            try to increase the regularization weight until it becomes stable. It is quite possible
            that optimizer might not work for problem scales.
            
In order to access the module externally please use following import commands:

from subset_select_entropic_FISTA_git import round_transpoly as round_transpoly
from subset_select_entropic_FISTA_git import feasible_proj as feasible_proj
from subset_select_entropic_FISTA_git import subset_select_FISTA as subset_select_FISTA 

Functions must be invoke in following order:
1. Run the subset_select_FISTA as subset_select_FISTA 
2. Run the feasible_proj
3. Run the round_transpoly

input/output details for each function are provided in corresponding documentation 
'''

#%%
def projection_simplex(V, z=1, axis=None):
    """
    This code is written by Mathieu Blondel and can be found at:
    https://gist.github.com/mblondel/c99e575a5207c76a99d714e8c6e08e89
    It is also part of the code for paper titled: Smooth and Sparse Optimal Transport
    paper can be found at: https://arxiv.org/abs/1710.06276
    and complete code for the paper can be found at: https://github.com/mblondel/smooth-ot

    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    axis: None or int
        axis=None: project V by P(V.ravel(); z)
        axis=1: project each V[i] by P(V[i]; z[i])
        axis=0: project each V[:, j] by P(V[:, j]; z[j])
    """
    if axis == 1:
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        z = np.ones(len(V)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        ind = np.arange(n_features) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / rho
        return np.maximum(V - theta[:, np.newaxis], 0)

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_simplex(V, z, axis=1).ravel()


#%%
def round_transpoly(F, r, c):
    '''
    implementation of ROUND function in Algorithm 1
    https://papers.nips.cc/paper/2017/file/491442df5f88c6aa018e86dac21d3606-Paper.pdf
    written by: Bilal Riaz bilalria@udel.edu
    inputs: 
    -- r : source marginal vector from probability simplex R^m
    -- c : target marginal vector from probability simplex R^n
    -- F : Positive matrix to be rounded onto transport polytope.
    outputs:
    -- A : Rounded version of matrix X.
    Note: This rounding function is written for the matrices which are output of exponentiation,
          therefore with non-zero rows and columns. For testing and use only input positive matrices. 
          It returns NAN terms for inputs with zeros rows or zero columns.

    '''

    assert np.isclose(np.sum(r), 1), "source points must lie on probability simplex."
    assert np.isclose(np.sum(c), 1), "target points must lie on probability simplex."

    r = r.reshape([np.size(r), 1])
    c = c.reshape([np.size(c), 1])

    A = F
    n = F.shape[1]

    # row sums
    r_A = np.sum(A, axis=1, keepdims=1)  # returns column vector
    #ratio_r = np.divide(r, r_A, out=np.zeros_like(r_A), where=~np.isclose(r_A,np.zeros_like(r_A), 1e-13, atol=1e-13))  # returns column vector
    # ratio_r = np.divide(r, r_A, out=np.full_like(r_A, np.inf, dtype=np.double), where=np.isclose(r_A, 0, atol = 1e-13)) # returns column vector
    ratio_r = np.divide(r, r_A, out=np.ones_like(r_A), where=r<r_A) # returns column vector
    #ratio_r = np.divide(r, r_A, out=np.ones_like(r_A), where=~(r_A==0))

    scaling_r = np.minimum(1, ratio_r)  # returns column vector # redundant line
    # row scaling
    A = scaling_r * A

    # column sum
    c_A = np.sum(A, axis=0, keepdims=1)  # returns row vector
    # ratio_c = np.divide(c.T, c_A, out=np.zeros_like(c_A), where=~np.isclose(c_A,np.zeros_like(c_A), 1e-13, atol=1e-13))  # returns row vector
    # ratio_c = np.divide(c.T, c_A, out=np.full_like(c_A, np.inf, dtype=np.double), where=np.isclose(c_A, 0, atol = 1e-13)) # returns row vector
    ratio_c = np.divide(c.T, c_A, out=np.ones_like(c_A), where=c.T<c_A)
    #ratio_c = np.divide(c.T, c_A, out=np.ones_like(c_A), where=~(c_A==0))
    scaling_c = np.minimum(1, ratio_c)   # returns row vector  # redundant line
    # column scaling
    A = scaling_c * A

    r_A = np.sum(A, axis=1, keepdims=1)  # returns column vector
    c_A = np.sum(A, axis=0, keepdims=1)  # returns row vector

    err_r = r - r_A
    err_c = c.T - c_A

    if (np.linalg.norm(err_r, ord=1)) == 0:
        return A
    else:
        A = A + err_r @ err_c / (np.linalg.norm(err_r, ord=1))
        return A

#%%
def feasible_proj(d, L):
    '''
    Feasibility projection for subset selection
    written by: Bilal Riaz (bilalria@udel.edu)

    solving the problem min_{x} norm(x-d, ord = 2) s.t. 0<=x<=1/L and sum(x) = 1 using FPDG (Beck chapter 12 Algorithm 4)

    inputs:
    -- d: a vector on probability simplex in \mathbb{R}^{n} which does not staisfy the subset selection constraint
    -- L : subset cardinality constraint parameter 1<=L<=n
    output:
    -- x: feasibility projection on to subset of probability simplex,
          such that each of the probabilities are upper bounded by 1/L

    '''
    
    Lt = 4  # step size chososen for faster convergence Lt must be greater than the number constraints
    d = d.squeeze()

    n = d.size
    b = 1/L * np.ones([n])

    # duals for each of the constraint
    # There us redundancy in constraints here, these additional constraints make the algorithm to converge faster.
    y1 = np.ones([n])
    y2 = np.ones([n])
    y3 = np.ones([n])

    w1 = y1
    w2 = y2
    w3 = y3

    t = 1
    while True:
        x = w1 + w2 + w3 + d
        
        y1_prev = y1
        y2_prev = y2
        y3_prev = y3

        # dual projection onto nonnegative orthant
        y1 = w1 - (1/Lt)*x + (1/Lt)*np.maximum(x - Lt*w1, 0)
        # dual projection onto inequality constraints
        y2 = w2 - (1/Lt)*x + (1/Lt) * np.minimum(x - Lt*w2, b)
        # dual projection onto to the probability simplex
        y3 = w3 - (1/Lt)*x + (1/Lt) * projection_simplex(x-Lt*w3)
        t_prev = t
        t = (1 + np.sqrt(1 + 4*t**2))/2

        w1 = y1 + ((t_prev - 1)/t)*(y1 - y1_prev)
        w2 = y2 + ((t_prev - 1)/t)*(y2 - y2_prev)
        w3 = y3 + ((t_prev - 1)/t)*(y3 - y3_prev)
        
        z = y1 + y2 + y3 + d
        
        if np.all(np.abs(z[z > 1/L] - 1/L) < 1e-15) and np.isclose(np.sum(z), 1, rtol=1e-14, atol=1e-14):
            break

    return z

#%%
def subset_select_FISTA(mu, L, K, M, gamma=1e-1, Lt=1e-1, eta = 1.1, max_iter = 20000, back_tracking = False, disp_iter=False):
    '''
    implementation of FISTA based algorithm for entropic regulzrized subset selection
    written by: Bilal Riaz bilalria@udel.edu
    inputs: 
    -- mu : source marginal vector from probability simplex R^m
    -- L : Thresholding parameter for target distribution
    -- M : Euclidian Distance matrix with each element raised to p, where p indicated the Wasserstein p-distance 
    -- Lt: step size
    -- gamma: positive regularization parameter (default: gamma =  1e-1)
    -- K : Gibbs Kernel Matrix defined as exp(-M/gamma)
    -- eta: scaling parameter for backtracking search, would be greater than 1 (default: eta = 1.1)
    -- back_tracking: Boolean parameter to choose between back-tracking and constant step-size (default: back_tracking = False)
    -- max_iter: maximum FISTA iterations (default: max_iter = 20000)
    -- disp_iter: Boolean variable to indicate whether to display error progress over iterations
    outputs:
    -- alpha_best: alpha_best  \in \mathbb{R}^{m} is the optimal dual variable corresponding equality constraint.
    -- beta_best: beta_best \in \mathbb{R}^{n}_{+} is dual variable corresponding inequality constraint.
    -- P_best:  P_best = np.exp(-(M + alpha_best + beta_best.T)/gamma)
    -- obj_vals: array of objective/cost values
    
    Note: Since the error for FISTA iterations is not necessarily monotonically decreasing, therefore we return alpha_best and beta_best
    '''

    m, n = K.shape
    assert m == mu.size, "dimensionality mismatch"
    assert ((L >= 1) and(L <= n)), "Mass-preservancy violated"
    assert np.isclose(np.sum(mu), 1), "source points must lie on probability simplex."

    mu = mu.reshape(mu.size, 1)
    nu = np.ones([n, 1])/L

    P = mu@nu.T
    beta = np.zeros([n, 1])
    alpha = np.zeros([m, 1])
    # beta = np.random.rand(n,1)
    # alpha = np.random.rand(m,1)
    
    # beta = beta/np.sum(beta)
    # alpha = alpha/np.sum(alpha)

    def objective(alpha, beta, gamma, mu, nu):
        temp1 = gamma * np.sum(np.exp(-1/gamma * (M + alpha + beta.T)))
        temp2 = np.sum(alpha*mu) + np.sum(beta*nu)
        return temp1 + temp2

    obj_vals = np.zeros([max_iter + 1])
    obj_vals[0] = objective(alpha, beta, gamma, mu, nu)

    obj_best = obj_vals[0]
    alpha_best = alpha.copy()
    beta_best = beta.copy()
    P_best = np.exp(-1/gamma * (M + alpha_best + beta_best.T))
    
    
    # s0 = np.sum(P_best.copy(), axis = 0, keepdims=1).T
    # gaps = np.zeros([s0.size, max_iter+1])
    # gaps[:,0] = np.squeeze(s0)
    
    if disp_iter is True:
        print("initial objective value")
        print(obj_vals[0])
    
    
    # The accelerated proximal gradeint method provably
    # converges if Lt = 1/gamma, but here we provide capability to use other values also
    # Lt = 1/gamma
    t = 1

    y = beta.copy()
    for k in range(0, max_iter):
        beta_prev = beta.copy()
        # alpha update
        alpha = - gamma * (np.log(mu) - np.log(K @ np.exp(-beta/gamma)))  
        # bidual gradient 
        grad_y = 1/L - np.exp(-y/gamma) * (K.T @ np.exp(-alpha/gamma))   

        # backtracking line search
        if back_tracking is True:
            L_prev = Lt
            fy = objective(alpha, y, gamma, mu, nu)
            z = y - 1/L_prev * grad_y
            z[z < 0] = 0
            while objective(alpha, z, gamma, mu, nu) > (fy + (grad_y.T@(z - y) + L_prev/2 * np.sum((z-y)**2)).item()):
                L_prev = L_prev * eta
                z = y - 1/L_prev * grad_y
                z[z < 0] = 0
            Lt = L_prev

        # beta update
        beta_0 = y - 1/Lt*grad_y  
        beta = beta_0.copy()
        beta[beta < 0] = 0
        
        # t  update
        t_prev = np.copy(t)
        t = (1 + np.sqrt(1 + 4*t**2))/2
        # bidual update
        y = beta + ((t_prev -1)/t) * (beta - beta_prev)
        
        # Auxiliary computations to return values

        obj_vals[k + 1] = objective(alpha, beta, gamma, mu, nu)
        obj_best = np.min([obj_vals[k+1], obj_best])
        
        #gaps[:,k] = np.squeeze(np.sum(P_best, axis = 0, keepdims=1).T -s0)
        if obj_vals[k+1] <= obj_best:
            alpha_best = alpha.copy()
            beta_best = beta.copy()
            P_best = np.exp(-1/gamma * (M + alpha_best + beta_best.T))
            
        if disp_iter is True:
            print("iter = ", k)
            print("objective value = ", obj_vals[k + 1])
        
        #gaps[:,k+1] = np.squeeze(np.sum(P_best.copy(), axis = 0, keepdims=1).T - s0)

    P_best = P_best/np.sum(P_best)
    
    # return P_best, obj_vals, alpha_best, beta_best, #
    
    P = np.exp(-(M+alpha+beta.T)/gamma)
    return P, obj_vals, alpha_best, beta_best

#%%

def subset_select_FISTA2(mu, L, K, M, gamma=1e-1, Lt=1e-1, eta = 1.1, max_iter = 20000, back_tracking = False, disp_iter=False):
    '''
    implementation of FISTA based algorithm for entropic regulzrized subset selection
    written by: Bilal Riaz bilalria@udel.edu
    inputs: 
    -- mu : source marginal vector from probability simplex R^m
    -- L : Thresholding parameter for target distribution
    -- M : Euclidian Distance matrix with each element raised to p, where p indicated the Wasserstein p-distance 
    -- Lt: step size
    -- gamma: positive regularization parameter (default: gamma =  1e-1)
    -- K : Gibbs Kernel Matrix defined as exp(-M/gamma)
    -- eta: scaling parameter for backtracking search, would be greater than 1 (default: eta = 1.1)
    -- back_tracking: Boolean parameter to choose between back-tracking and constant step-size (default: back_tracking = False)
    -- max_iter: maximum FISTA iterations (default: max_iter = 20000)
    -- disp_iter: Boolean variable to indicate whether to display error progress over iterations
    outputs:
    -- alpha_best: alpha_best  \in \mathbb{R}^{m} is the optimal dual variable corresponding equality constraint.
    -- beta_best: beta_best \in \mathbb{R}^{n}_{+} is dual variable corresponding inequality constraint.
    -- P_best:  P_best = np.exp(-(M + alpha_best + beta_best.T)/gamma)
    -- obj_vals: array of objective/cost values
    
    Note: Since the error for FISTA iterations is not necessarily monotonically decreasing, therefore we return alpha_best and beta_best
    '''

    m, n = K.shape
    assert m == mu.size, "dimensionality mismatch"
    assert (L >= 1 and L <= n), "Mass-preservancy violated"
    assert np.isclose(np.sum(mu), 1), "source points must lie on probability simplex."

    mu = mu.reshape(mu.size, 1)
    nu = np.ones([n, 1])/L

    P = mu@nu.T
    # beta = np.ones([n, 1])
    # alpha = np.ones([m, 1])
    beta = np.random.rand(n,1)
    alpha = np.random.rand(m,1)
    
    beta = beta/np.sum(beta)
    alpha = alpha/np.sum(alpha)

    def objective(alpha, beta, gamma, mu, nu):
        temp1 = gamma * np.sum(np.exp(-1/gamma * (M + alpha + beta.T)))
        temp2 = np.sum(alpha*mu) + np.sum(beta*nu)
        return temp1 + temp2

    obj_vals = np.zeros([max_iter + 1])
    obj_vals[0] = objective(alpha, beta, gamma, mu, nu)

    obj_best = obj_vals[0]
    alpha_best = alpha.copy()
    beta_best = beta.copy()
    P_best = np.exp(-1/gamma * (M + alpha_best + beta_best.T))
    
    
    # s0 = np.sum(P_best.copy(), axis = 0, keepdims=1).T
    # gaps = np.zeros([s0.size, max_iter+1])
    # gaps[:,0] = np.squeeze(s0)
    
    if disp_iter == True:
        print("initial objective value")
        print(obj_vals[0])
    
    
    # The accelerated proximal gradeint method provably
    # converges if Lt = 1/gamma, but here we provide capability to use other values also
    # Lt = 1/gamma
    t = 1

    y = beta.copy()
    for k in range(0, max_iter):
        beta_prev = beta.copy()
        # alpha update
        alpha = - gamma * (np.log(mu) - np.log(K @ np.exp(-beta/gamma)))  
        # bidual gradient 
        grad_y = 1/L - np.exp(-y/gamma) * (K.T @ np.exp(-alpha/gamma))   

        # backtracking line search
        if back_tracking == True:
            L_prev = Lt
            fy = objective(alpha, y, gamma, mu, nu)
            z = y - 1/L_prev * grad_y
            z[z < 0] = 0
            while objective(alpha, z, gamma, mu, nu) > (fy + (grad_y.T@(z - y) + L_prev/2 * np.sum((z-y)**2)).item()):
                L_prev = L_prev * eta
                z = y - 1/L_prev * grad_y
                z[z < 0] = 0
            Lt = L_prev
            
        # beta update
        beta_0 = y - 1/Lt*grad_y  
        z = beta_0.copy()
        z[z < 0] = 0
        
        if objective(alpha, z, gamma, mu, nu) > objective(alpha, beta_prev, gamma, mu, nu):
            beta = beta_prev.copy()
        else:
            beta = z.copy()
            
        
        # t  update
        t_prev = np.copy(t)
        t = (1 + np.sqrt(1 + 4*t**2))/2
        # bidual update
        # y = beta + ((t_prev -1)/t) * (beta - beta_prev) 
        y = beta + (t_prev/t)*(z - beta) + ((t_prev -1)/t) * (beta - beta_prev) 
        
        # Auxiliary computations to return values

        obj_vals[k + 1] = objective(alpha, beta, gamma, mu, nu)
        obj_best = np.min([obj_vals[k+1], obj_best])
        
        #gaps[:,k] = np.squeeze(np.sum(P_best, axis = 0, keepdims=1).T -s0)
        if obj_vals[k+1] <= obj_best:
            alpha_best = alpha.copy()
            beta_best = beta.copy()
            P_best = np.exp(-1/gamma * (M + alpha_best + beta_best.T))
            
        if disp_iter == True:
            print("iter = ", k)
            print("objective value = ", obj_vals[k + 1])
        
        #gaps[:,k+1] = np.squeeze(np.sum(P_best.copy(), axis = 0, keepdims=1).T - s0)

    P_best = P_best/np.sum(P_best)
    
    # return P_best, obj_vals, alpha_best, beta_best, #
    
    P = np.exp(-(M+alpha+beta.T)/gamma)
    return P, obj_vals, alpha_best, beta_best
# %%

def subset_select_FISTA3(mu, L, K, M, gamma=1e-1, Lt=1e-1, eta = 1.1, max_iter = 20000, back_tracking = False, disp_iter=False):
    '''
    implementation of FISTA based algorithm for entropic regulzrized subset selection
    written by: Bilal Riaz bilalria@udel.edu
    inputs: 
    -- mu : source marginal vector from probability simplex R^m
    -- L : Thresholding parameter for target distribution
    -- M : Euclidian Distance matrix with each element raised to p, where p indicated the Wasserstein p-distance 
    -- Lt: step size
    -- gamma: positive regularization parameter (default: gamma =  1e-1)
    -- K : Gibbs Kernel Matrix defined as exp(-M/gamma)
    -- eta: scaling parameter for backtracking search, would be greater than 1 (default: eta = 1.1)
    -- back_tracking: Boolean parameter to choose between back-tracking and constant step-size (default: back_tracking = False)
    -- max_iter: maximum FISTA iterations (default: max_iter = 20000)
    -- disp_iter: Boolean variable to indicate whether to display error progress over iterations
    outputs:
    -- alpha_best: alpha_best  \in \mathbb{R}^{m} is the optimal dual variable corresponding equality constraint.
    -- beta_best: beta_best \in \mathbb{R}^{n}_{+} is dual variable corresponding inequality constraint.
    -- P_best:  P_best = np.exp(-(M + alpha_best + beta_best.T)/gamma)
    -- obj_vals: array of objective/cost values
    
    Note: Since the error for FISTA iterations is not necessarily monotonically decreasing, therefore we return alpha_best and beta_best
    '''

    m, n = K.shape
    assert m == mu.size, "dimensionality mismatch"
    assert (L >= 1 and L <= n), "Mass-preservancy violated"
    assert np.isclose(np.sum(mu), 1), "source points must lie on probability simplex."

    mu = mu.reshape(mu.size, 1)
    nu = np.ones([n, 1])/L

    P = mu@nu.T
    beta = np.ones([n, 1])
    alpha = np.ones([m, 1])
    # beta = np.random.rand(n,1)
    # alpha = np.random.rand(m,1)
    
    beta = beta/np.sum(beta)
    alpha = alpha/np.sum(alpha)

    def objective(alpha, beta, gamma, mu, nu):
        temp1 = gamma * np.sum(np.exp(-1/gamma * (M + alpha + beta.T)))
        temp2 = np.sum(alpha*mu) + np.sum(beta*nu)
        return temp1 + temp2

    obj_vals = np.zeros([max_iter + 1])
    obj_vals[0] = objective(alpha, beta, gamma, mu, nu)

    obj_best = obj_vals[0]
    alpha_best = alpha.copy()
    beta_best = beta.copy()
    P_best = np.exp(-1/gamma * (M + alpha_best + beta_best.T))
    
    
    # s0 = np.sum(P_best.copy(), axis = 0, keepdims=1).T
    # gaps = np.zeros([s0.size, max_iter+1])
    # gaps[:,0] = np.squeeze(s0)
    
    if disp_iter == True:
        print("initial objective value")
        print(obj_vals[0])
    
    
    # The accelerated proximal gradeint method provably
    # converges if Lt = 1/gamma, but here we provide capability to use other values also
    # Lt = 1/gamma
    t = 1
    
    x = np.vstack((alpha, beta))
    y = beta.copy()
    mu_nu = np.vstack((mu, 1/L*np.ones([n,1])))
    
    for k in range(0, max_iter):
        beta_prev = beta.copy()
        # alpha update
        # alpha = - gamma * (np.log(mu) - np.log(K @ np.exp(-beta/gamma)))  
        # # bidual gradient 
        # grad_y = 1/L - np.exp(-y/gamma) * (K.T @ np.exp(-alpha/gamma)) 
        x_prev = x.copy()  
        Z = np.exp(-(M + y[0:m] + y[-n:].T )/gamma)
        grad_y = mu_nu - np.vstack((np.sum(Z, axis = 1, keepdims = 1), np.sum(Z, axis=0, keepdims = 1).T))
        x0 = x - 1/Lt * grad_y
        x0[-n:] = np.minimum(x0[-n:],0)
        x = x0.copy()
        t_prev = np.copy(t)
        t = (1 + np.sqrt(1 + 4*t**2))/2
        y = x + ((t_prev -1)/t)*(x - x_prev)
        
        # Auxiliary computations to return values

        obj_vals[k + 1] = objective(x[0:m], x[-n:], gamma, mu, nu)
        obj_best = np.min([obj_vals[k+1], obj_best])
        
        #gaps[:,k] = np.squeeze(np.sum(P_best, axis = 0, keepdims=1).T -s0)
        if obj_vals[k+1] <= obj_best:
            alpha_best = x[0:m].copy()
            beta_best = x[-n:].copy()
            P_best = np.exp(-1/gamma * (M + alpha_best + beta_best.T))
            
        if disp_iter == True:
            print("iter = ", k)
            print("objective value = ", obj_vals[k + 1])
        
        #gaps[:,k+1] = np.squeeze(np.sum(P_best.copy(), axis = 0, keepdims=1).T - s0)

    P_best = P_best/np.sum(P_best)
    
    # return P_best, obj_vals, alpha_best, beta_best, #
    
    P = np.exp(-(M+alpha+beta.T)/gamma)
    return P_best, obj_vals, alpha_best, beta_best

# %%
