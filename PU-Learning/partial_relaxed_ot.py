import time
import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
import ot

from ot.optim import semirelaxed_cg as s_ot
import utils


def p_rot(mu, nu, M, penalty, reg):
    
    assert len(mu) == M.shape[0], "shape mismatch"
    assert len(nu) == M.shape[1], "shape mismatch"
    assert np.isclose(np.sum(mu), 1), "mu must lie in probability simplex"
    assert np.isclose(np.sum(mu), 1), "mu must lie in probability simplex"
    
    if penalty == "l1":
        def f(P):
            return np.linalg.norm(np.sum(P, axis=0, keepdims=1).T - nu.reshape(-1,1), 1)
            
        def df(P):
            grad = np.repeat(np.sign(np.sum(P,axis=0, keepdims=1) - nu.reshape([1, len(nu)])), repeats=len(mu), axis=0)
            return grad 
        
        P_star = s_ot(mu, nu, M, reg, f, df, numItermax=200)
        return P_star
    
    if penalty == "l2":
        def f(P):
            return (np.linalg.norm(np.sum(P, axis=0, keepdims=1).T - nu.reshape(-1,1), 2))**2
            
        def df(P):
            grad = 2 * np.repeat(np.sum(P, axis=0, keepdims=1) - nu.reshape([1, len(nu)]), repeats=len(mu), axis=0)
            return grad 

        P_star = s_ot(np.squeeze(mu), np.squeeze(nu), M, reg, f, df, numItermax=20)
        return P_star
        
        
def compute_partial_reg_ot(dataset_p, dataset_u, n_pos, n_unl, prior, nb_reps, penalty, reg):
    
    y_hat_list = []
    y_u_list = []
    nu_star_list = []
    P_list = []
    
    
    # transp_rpot_group_list = []
    mu = np.ones([n_pos])/n_pos
    nu = np.ones([n_unl])/n_unl
    
    for i in range(nb_reps):
        P, U, y_u = utils.draw_p_u_dataset_scar(dataset_p, dataset_u, n_pos, n_unl, prior, i)  # seed=i
        y_u_list.append(y_u.to_numpy())
        
        Ctot = np.linalg.norm(np.expand_dims(P.to_numpy(), axis=1) - U.to_numpy(), ord=2, axis=2)
        
        nb_unl_pos = int(np.sum(y_u))

        P = p_rot(mu, nu, Ctot, penalty, reg)
        P_list.append(P)
        
        y_hat = np.zeros(len(y_u))
        nu_star = np.sum(P, axis=0)
        y_hat[np.argsort(-nu_star)[0:nb_unl_pos]] = 1
        
        y_hat_list.append(y_hat)
        nu_star_list.append(nu_star)
        
        
    return P_list, y_u_list, y_hat_list, nu_star_list

        
            
            
        
            
            
    
                
        
            
    
    







