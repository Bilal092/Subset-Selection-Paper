import numpy as np

def subset_select_ipot(X, Y, mu, nu, c,  gamma, max_outer_iter, max_inner_iter = 20 , wd = 2, disp_iter = False, return_map = False):
    '''
    implementation of FISTA based algorithm for approximating exact Wasserstein DIstance
    written by: Bilal Riaz bilalria at udel dot edu / bilalriaz at live dot kom
    inputs:
    -- X: Matrix of d dimensional target data points: shape m x d
    -- Y: Matrix of d dimensional source data points: shape n x d
    -- mu : target marginal vector from probability simplex R^m
    -- L : Thresholding parameter for target distribution
    -- max_outer_iter: maximum number of proximal point evaluations
    -- max_inner_iter: maxiumum number of iterations in FISTA based evaluation of proximal point.
    -- gamma: positive proximal thresholding parameter
    -- disp_iter: Boolean variable to indicate whether to display error progress over iterations
    outputs:
    -- costs: A list of primal cost evaluated accross iterations.
    -- P: suport subset selection plan. 
    
    Note: Since the error for FISTA iterations is not necessarily monotonically decreasing, therefore we return alpha_best and beta_best
    '''
    
    M = np.linalg.norm(X[:,np.newaxis,:] - Y, axis=2)**wd
    M = M/np.max(M)
    K = np.exp(- M /gamma)
    m, n = K.shape
    
    assert m == mu.size, "dimensionality mismatch"
    assert n == nu.size, "dimensionality mismatch"
    
    assert ((c >= 1)), "Mass-preservancy violated"
    assert np.isclose(np.sum(mu), 1), "target points must lie on probability simplex."
    assert np.isclose(np.sum(nu), 1), "source points must lie on probability simplex."
    
    def objective(alpha, beta, gamma, mu):
        temp1 = gamma * np.sum(np.exp(-1/gamma * (M + alpha + beta.T)))
        temp2 = np.sum(alpha*mu) + c*np.sum(beta*nu)
        return temp1 + temp2
    
    mu = mu.reshape([m,1])
    nu = nu.reshape([n,1])
    P = np.ones_like(K)/(m*n)
    alpha = np.zeros([m,1])
    beta = np.zeros([n,1])
    costs = []
    Lt = 1/gamma
    
    obj_vals = np.zeros([max_outer_iter + 1])
    obj_vals[0] = objective(alpha, beta, gamma, mu)
    
    obj_best = obj_vals[0]
    alpha_best = alpha.copy()
    beta_best = beta.copy()
    P_best = np.exp(-1/gamma * (M + alpha_best + beta_best.T))
    
    if disp_iter is True:
        print("initial objective value")
        print(obj_vals[0])
    
    for outer_iter in range(0, max_outer_iter):
        Q = K*P
        y = beta_best.copy()
        t = 1
        for inner_iter in range(0, max_inner_iter):
            beta_prev = beta.copy()
            
            # alpha update
            alpha = gamma*(np.log(Q@np.exp(-beta/gamma)) - np.log(mu))
            grad_y = c*nu - np.exp(-y/gamma) * (Q.T @ np.exp(-alpha/gamma))   
            beta_0 = y - 1/Lt * grad_y
            beta = beta_0.copy()
            beta[beta<0] = 0
            
            t_prev = np.copy(t)
            t = (1 + np.sqrt(1 + 4*t**2))/2
            # bidual update
            y = beta + ((t_prev -1)/t) * (beta - beta_prev)
            
            alpha_best = alpha.copy()
            beta_best = beta.copy()
            
            obj = objective(alpha, beta, gamma, mu)
            
            # obj = objective(alpha, beta, gamma, mu)
            # obj_best = np.min([obj, obj_best])
            
            # if obj <= obj_best:
            #     alpha_best = alpha.copy()
            #     beta_best = beta.copy()   
        
        P = np.diag(np.exp(-alpha_best.squeeze()/gamma)) @ (Q@np.diag(np.exp(-beta_best.squeeze()/gamma)))
        costs.append(np.sum(P*M))
        
        if disp_iter == True:
            print("iter = ", outer_iter)
            print("cost = ", obj)
    
    if return_map == True:
        return costs, P, alpha, beta
    else:
        return costs