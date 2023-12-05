import torch

    
def subset_select_ipot_pytorch(X, Y, mu, nu, c,  gamma, max_outer_iter, max_inner_iter = 20 , wd = 2, disp_iter = False, return_map = False):
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
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = "cpu"

    M = torch.norm(X.unsqueeze(1) - Y, dim=2)**wd
    M = M/torch.max(M)
    K = torch.exp(- M /gamma)
    m, n = K.shape
    mu = mu.reshape([m,1])
    nu = nu.reshape([n,1])

    assert m == len(mu), "dimensionality mismatch"
    assert n == len(nu), "dimensionality mismatch"
    assert c >= 1, "Mass-preservancy violated"
    assert torch.isclose(torch.sum(mu), torch.tensor([1.0], device=device)), "target points must lie on probability simplex."
    assert torch.isclose(torch.sum(nu), torch.tensor([1.0], device=device)), "source points must lie on probability simplex."

    def objective(alpha, beta, gamma, mu, nu, c):
        temp1 = gamma * torch.sum(torch.exp(-1/gamma * (M + alpha + beta.T)))
        temp2 = torch.sum(alpha*mu) + c * torch.sum(beta*nu)
        return (temp1 + temp2).item()

    P = torch.ones_like(K)/(m*n)
    alpha = torch.zeros([m,1], dtype=X.dtype, device=device)
    beta = torch.zeros([n,1], dtype=X.dtype, device = device)
    costs = []
    Lt = torch.tensor(1/gamma).to(device)
    obj_vals = torch.zeros([max_outer_iter + 1])
    obj_vals[0] = objective(alpha, beta, gamma, mu, nu, c)
    obj_best = obj_vals[0]
    alpha_best = alpha.clone()
    beta_best = beta.clone()
    P_best = torch.exp(-1/gamma * (M + alpha_best + beta_best.T))

    if disp_iter is True:
        print("initial objective value")
        print(obj_vals[0].item())

    for outer_iter in range(0, max_outer_iter):
        Q = K*P
        y = beta_best.clone()
        t = torch.tensor([1.0], device=device)
        for inner_iter in range(0, max_inner_iter):
            beta_prev = beta.clone()

            # alpha update
            alpha = gamma*(torch.log(Q@torch.exp(-beta/gamma)) - torch.log(mu))
            grad_y = c*nu - torch.exp(-y/gamma) * (Q.T @ torch.exp(-alpha/gamma))
            beta_0 = y - 1/Lt * grad_y
            beta = beta_0.clone()
            beta[beta<0] = 0
            t_prev = t.clone()
            t = (1 + torch.sqrt(1 + 4*t**2))/2
            # bidual update
            y = beta + ((t_prev -1)/t) * (beta - beta_prev)
            alpha_best = alpha.clone()
            beta_best = beta.clone()
            obj = objective(alpha, beta, gamma, mu, nu, c)

            # obj = objective(alpha, beta, gamma, mu)
            # obj_best = np.min([obj, obj_best])

            # if obj <= obj_best:
            #     alpha_best = alpha.copy()
            #     beta_best = beta.copy()
        P = torch.diag(torch.exp(-alpha_best.squeeze()/gamma)) @ (Q@torch.diag(torch.exp(-beta_best.squeeze()/gamma)))
        costs.append(torch.sum(P*M).item())

        if disp_iter == True:
            print("iter = ", outer_iter)
            print("cost = ", obj)

    if return_map == True:
        return costs, P, alpha, beta
    else:
        return costs

