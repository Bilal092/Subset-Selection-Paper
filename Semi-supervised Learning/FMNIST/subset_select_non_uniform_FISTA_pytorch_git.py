import torch
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = "cpu"
def subset_select_non_uniform_FISTA_pytorch(mu, nu, c, K, M, gamma=1e-1, Lt=1e-1, eta = 1.1, max_iter = 20000, back_tracking = False, disp_iter=False):
    '''
    pytorch implementation of FISTA based algorithm for entropic regulzrized subset selection
    written by: Bilal Riaz bilalria@udel.edu
    inputs:
    -- mu : target marginal vector from probability simplex R^m.
    -- nu : source marginal vector from probability simplex R^n.
    -- c : Thresholding parameter for target distribution
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
    assert m == len(mu), "dimensionality mismatch."
    assert n == len(nu), "dimensionality mismatch."
    assert torch.isclose(torch.sum(mu), torch.tensor([1.0], device=device)), "target points must lie on probability simplex."
    assert torch.isclose(torch.sum(nu), torch.tensor([1.0], device=device)), "source points must lie on probability simplex."

    mu = mu.reshape([len(mu), 1])
    nu = nu.reshape([len(nu), 1])
    P = torch.ones_like(K)/(m*n)
    alpha = torch.zeros([m,1], dtype=torch.double, device=device)
    beta = torch.zeros([n,1], dtype=torch.double, device=device)

    def objective(alpha, beta, gamma, mu, nu, c):
        temp1 = gamma * torch.sum(torch.exp(-1/gamma * (M + alpha + beta.T)))
        temp2 = torch.sum(alpha*mu) + c * torch.sum(beta*nu)
        return (temp1 + temp2).item()

    obj_vals = torch.zeros([max_iter + 1])
    obj_vals[0] = objective(alpha, beta, gamma, mu, nu, c)
    obj_best = obj_vals[0]
    alpha_best = alpha.clone()
    beta_best = beta.clone()
    P_best = torch.exp(-1/gamma * (M + alpha_best + beta_best.T))

    if disp_iter == True:
        print("initial objective value")
        print(obj_vals[0])
    # The accelerated proximal gradeint method provably
    # converges if Lt = 1/gamma, but here we provide capability to use other values also
    # Lt = 1/gamma
    t = torch.tensor([1.0], device=device)
    y = beta.clone()

    for k in range(0, max_iter):
        beta_prev = beta.clone()
        # alpha update
        alpha = - gamma * (torch.log(mu) - torch.log(K @ torch.exp(-beta/gamma)))
        # bidual gradient
        grad_y = c*nu - torch.exp(-y/gamma) * (K.T @ torch.exp(-alpha/gamma))
        # backtracking line search
        if back_tracking == True:
            L_prev = Lt
            fy = objective(alpha, y, gamma, mu, nu,c)
            z = y - 1/L_prev * grad_y
            z[z < 0] = 0
            while objective(alpha, z, gamma, mu, nu, c) > (fy + (grad_y.T@(z - y) + L_prev/2 * torch.sum((z-y)**2)).item()):
                L_prev = L_prev * eta
                z = y - 1/L_prev * grad_y
                z[z < 0] = 0
            Lt = L_prev
        # beta update
        beta_0 = y - 1/Lt*grad_y
        beta = beta_0.clone()
        beta[beta < 0] = 0
        # t  update
        t_prev = torch.clone(t)
        t = (1 + torch.sqrt(1 + 4*t**2))/2
        # bidual update
        y = beta + ((t_prev -1)/t) * (beta - beta_prev)
        # Auxiliary computations to return values
        obj_vals[k + 1] = objective(alpha, beta, gamma, mu, nu, c)
        obj_best = torch.min(torch.tensor([obj_vals[k+1], obj_best])).item()
        alpha = alpha.clone().double()
        beta = beta.clone().double()

        if obj_vals[k+1] <= obj_best:
            alpha_best = alpha.clone()
            beta_best = beta.clone()
            P_best = torch.exp(-1/gamma * (M + alpha_best + beta_best.T))
        if disp_iter == True:
            print("iter = ", k)
            print("objective value = ", obj_vals[k + 1])
            print(torch.sum(torch.exp(-(M+alpha+beta.T)/gamma)))
    # P_best = P_best/torch.sum(P_best)
    P = torch.exp(-(M+alpha+beta.T)/gamma)
    return P, obj_vals, alpha_best, beta_best

