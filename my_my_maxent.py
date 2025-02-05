import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import scipy

import sys 
# sys.path.append('/oak/stanford/orgs/simes/rebjin/dqmc-dev/util')
sys.path.append('/Users/rebekahjin/Documents/Devereaux Group/dqmc-dev/util')
import util

from scipy.interpolate import CubicSpline
import math

import matplotlib.pyplot as plt
default_figsize = plt.rcParams['figure.figsize']

def maxent(G, K, m, opt_method='Bryan', constr_matrix=None, constr_vec=None, smooth_al=False, als=np.logspace(8, 1, 1+20*(8-1)), inspect_al=False, inspect_opt=False):
    """MaxEnt method to calculate A(w) for G(tau)=K(tau, w)*A(w) by maximizing Q[A(w); al]=al*S-0.5*chi^2.

    Args:
        G (array): Imaginary time correlator data (1xL). K (array): Kernel matrix (LxN). m (array): Model function (1xN).
        opt_method (str): Optimization method used to maximize Q. Options are:
            - 'Bryan': Bryan's method (only unconstrained).
            - 'cvxpy': Use cvxpy solver (constrained if constr_matrix and constr_vec provided, otherwise unconstrained).
        constr_matrix (array, optional): Constraint matrix B (MxN) for linear constraints B*A=b (Default: None).
        constr_vec (array, optional): Constraint vector b (Mx1) for linear constraints B*A=b (Default: None).
        als (array): Array of alpha values used in optimal alpha selection.
        inspect (bool): Whether to plot/print checks.
    Returns:
        A (array): Spectral function A(w) (1xN).
    """
    N = K.shape[1]
    L = K.shape[0]
    nbin = G.shape[0]
    Gavg = G.mean(0)

    # ---------- Change to basis in which C is diagonal (apparently makes calculating chi^2 easier?) ----------
    # sigma, Uc = np.linalg.eigh(np.cov(G.T) / nbin) # COB matrix
    # Uc = Uc.T
    # W = 1.0/sigma
    sigma, Uc = np.linalg.svd(G - Gavg, False)[1:]
    W = (nbin*(nbin-1)) / (sigma * sigma)
    # Recommended step: check C eigenvalues, too small (W too big) "might make optimizing Q[A] difficult"
    W_ratio_max = 1e8
    W_cap = W_ratio_max*W.min()
    n_large = np.sum(W.max() > W_cap)
    if W.max() > W_cap:
        # print(f"clipping {n_large} W values to W.min()*{W_ratio_max}")
        W[W > W_cap] = W_cap # Set values of W above W_cap to W_cap
    Kp = np.dot(Uc, K)
    Gavgp = np.dot(Uc, Gavg)
    
    # ---------- Select optimal al ----------
    # if opt_method=='cvxpy':
    #     als = np.logspace(7, 3, 160*4//8)
    al, As, chi2s = select_al(Gavgp, Kp, m, W, als, smooth=smooth_al, opt_method=opt_method, constr_matrix=constr_matrix, constr_vec=constr_vec, inspect_al=inspect_al, inspect_opt=inspect_opt)

    # ---------- Calculate A with optimal al ----------
    if opt_method == 'Bryan':
        A, _ = find_A_Bryan(Gavgp, Kp, m, W, al, inspect=inspect_opt)   #### issue with u_init here, come back to it
    elif opt_method == 'cvxpy':
        A = find_A_cvxpy(Gavgp, Kp, m, W, al, constr_matrix=constr_matrix, constr_vec=constr_vec, inspect=inspect_opt)
    else:
        raise ValueError(f"Invalid opt_method: '{opt_method}'. Expected 'Bryan' or 'cvxpy'.")

    return A, al, As, chi2s
    
def select_al(G, K, m, W, als, opt_method="Bryan", smooth=False, constr_matrix=None, constr_vec=None, inspect_al=False, inspect_opt=False):
    """Selects optimal alpha using BT method. 
    
    BT method calculates chi2 for optimized spectrum A* for every al in als.
    Returns optimal al as value in als with highest curvature in log-log plot of chi2 vs. al
    (idea is to get a good match to the data without overfitting).
    
    Args:
        G (array): Imaginary time correlator data (1xL). K (array): Kernel matrix (LxN). m (array): Model function (1xN).
        W (array): eigenvalues of .
        als (array): Array of alpha values over which to maximize Q.
        opt_method (str): Optimization method used to maximize Q. Options are:
            - 'Bryan': Bryan's method.
            - 'cvxpy': Convex optimization method.
        constr_matrix (array, optional): Constraint matrix B (MxN) for linear constraint B*A=b (Default: None).
        constr_vec (array, optional): Constraint vector b (Mx1) for linear constraint B*A=b (Default: None).
        inspect : asdf
    Returns:
        al (float): Optimal alpha value.
    """
    As = np.zeros((als.shape[0], K.shape[1]))
    Qs = np.zeros_like(als)
    Ss = np.zeros_like(als)
    chi2s = np.zeros_like(als)

    ### Precalculate SVD matrices for Bryan's method
    if opt_method == "Bryan":
        svd_threshold = 1e-12   # consider singular values less than threshold 0
        V, Sigma, U = np.linalg.svd(K, False)
        mask = (Sigma/Sigma.max() >= svd_threshold) # drop singular values less than threshold
        U = U.T[:, mask]
        SigmaVT = (V[:, mask] * Sigma[mask]).T
        M = np.dot(SigmaVT * W, SigmaVT.T)
        precalc = (U, SigmaVT, M)
        # Useful constants
        N = K.shape[1]
        s = M.shape[0]
        us = np.zeros((als.shape[0], M.shape[0]))

    ### Calculate Q, S, chi2 for all alphas in als
    for i, al in enumerate(als):
        if opt_method == 'Bryan':
            u_init = us[i-1]
            # config = {'mu_min': al/4.0, 'mu_max': al*1e100, 'mu_init': al}
            As[i], us[i] = find_A_Bryan(G, K, m, W, al, u_init=u_init, precalc=precalc, inspect=inspect_opt)
        elif opt_method == "cvxpy": 
            try:
                As[i] = find_A_cvxpy(G, K, m, W, al, constr_matrix=constr_matrix, constr_vec=constr_vec, inspect=inspect_opt)
            except Exception as e:
                print(f"find_A_cvxpy failed for {al:.2e} with error: {e}")
                As[i] = np.full(K.shape[1], np.nan) # Make array of nans if the optimization fails
            # As[i] = find_A_cvxpy(G, K, m, W, al, constr_matrix=constr_matrix, constr_vec=constr_vec, inspect=inspect_opt)
        Qs[i], Ss[i], chi2s[i] = Q(As[i], G, K, m, W, al, return_all=True) # these are nan too if A has nan

    ### Select optimal alpha based on curvature of log-log plot of chi2 vs. al
    # If constrained, make spline fit smoother bc chi2 is much more noisy
    # And choose curvature peak occurring at smallest al (not necessarily max)
    
    # Filter out nans
    valid_indices = ~np.isnan(chi2s)
    valid_als = als[valid_indices]
    valid_chi2s = chi2s[valid_indices]
    order = valid_als.argsort()
    if smooth:
        # Smooth modified BT, currently for use with constrained xy data
        # print('smooth BT')
        fit = scipy.interpolate.make_smoothing_spline(np.log(valid_als[order]), np.log(valid_chi2s[order]), lam=0.5)
        k = fit(np.log(als), 2)/(1 + fit(np.log(als), 1)**2)**1.5
        # k = fit(np.log(als), 2)/(1 + fit(np.log(als), 1)**2)**100
        k_range = max(k)-min(k)
        result = scipy.signal.find_peaks(k, prominence=k_range/5)
        peaks = result[0]
        al_idx = peaks[-1]
    else:
        # Default BT
        # Actually jk I'm smoothing it out a tiny bit and let's see
        fit = scipy.interpolate.make_smoothing_spline(np.log(valid_als[order]), np.log(valid_chi2s[order]), lam=0.02)
        # fit = CubicSpline(np.log(als[order]), np.log(chi2s[order])) 
        k = fit(np.log(als), 2)/(1 + fit(np.log(als), 1)**2)**1.5
        al_idx = k.argmax()
    al = als[al_idx]

    # inspect=False
    # if math.floor(math.log(al, 10)) != 6 and opt_method == 'cvxpy':
    #     inspect=True
    ### Optional plots for debugging
    if inspect_al:
        # Plot chi2 vs. al showing al selection and spline fit, with second derivative peaks.
        fig, ax = plt.subplots(ncols=2, figsize=(default_figsize[0]*2/1.2, default_figsize[1]/1.2), layout='constrained')
        ax[0].scatter(als, chi2s, s=1.5)
        ax[0].loglog(als, np.exp(fit(np.log(als))), color='r', label='f')
        ax[0].set_xlabel(r"$\alpha$")
        ax[0].set_ylabel(r"$\chi^2$")
        ax[0].axvline(al, color='g', label = rf"$\alpha$ = {np.round(al, 2)}")
        ax[0].annotate(rf"$\alpha$ = {np.round(al, 2)}", (0.05, 0.9), xycoords='axes fraction', fontsize=10, color='g')

        ax[1].plot(als, k)
        ax[1].set_xscale("log")
        ax[1].set_ylabel(r"$f''/(1 + f'^2)^{1.5}$")
        # ax[1].plot(als, fit(np.log(als), 2)) # Plot 2nd derivative directly
        # ax[1].plot(als, fit(np.log(als), 1)) # Also plot 1st derivative
        if smooth:
            ax[1].scatter(als[peaks], k[peaks], s=5)
            ax[1].scatter(als[i], k[i], color='g', s=5)
        
        # Plot Q, S, and chi2.
        xlim=(0, 10**6)
        fig, ax = plt.subplots(ncols=3, figsize=(default_figsize[0]*3/1.2, default_figsize[1]/1.2), layout='constrained')
        plot_list = [chi2s, Qs, Ss]
        plot_labels = [r"$\chi^2$", r"$Q$", r"$S$"]
        for i in range(3):
            ax[i].scatter(als, plot_list[i])
            ax[i].set_xscale("log")
            ax[i].set_xlabel(r"$\alpha$")
            ax[i].set_ylabel(plot_labels[i])
            ax[i].axvline(al, color='g')
        ax[0].annotate(rf"$\alpha$ = {np.round(al, 2)}", (0.05, 0.9), xycoords='axes fraction', fontsize=10, color='g')
        ax[0].set_yscale("log")
        plt.show()

    return al, As, chi2s


    
def find_A_Bryan(G, K, m, W, al, u_init=None, precalc=None, inspect=False):
    """Calculate A for given alpha using Bryan's optimization algorithm.

    Bryan's algorithm optimizes Q over a smaller singular space using unconstrained Newton's method (with Marquardt-Levenberg),
    finding optimal u* where A*=m exp(Uu*). Adapted directly from Edwin's code.

    Args:
        G (array): Imaginary time correlator data (1xL). K (array): Kernel matrix (LxN). m (array): Model function (1xN).
        W (array): ...
        al (float): Fixed alpha value.
        u_init (array, optional): Initial guess for optimizer.
        precalc (tuple, optional): Precomputed SVD matrices of K.
        return_u (bool, optional): Whether to also return u*.
    Returns:
        A (array): Optimization result, spectral function A(w) (1xN).
    """
    ### Solver settings
    mu_multiplier = 2.0  # increase/decrease mu by multiplying/dividing by this
    mu_min, mu_max = al/4.0, al*1e100  # range of nonzero mu
    step_max_accept = 0.5  # maximum size of an accepted step
    step_drop_mu = 0.125  # decrease mu if step_size < this
    dQ_threshold = 1e-10
    max_small_dQ = 7  # stop if dQ/Q < dQ_threshold this many times in a row
    max_iter = 1000  # max num of iterations if above condition not met

    ### Grad and hess funcs
    def grad_Q(u, G, K, m, W, al, precalc):
        '''Gradient of Q w.r.t. u = alpha u + g'''
        U, SigmaVT, M = precalc
        A = m*np.exp(U@u)
        return -(al*u + SigmaVT@((K@A-G)*W))
    def hess_Q(u, G, K, m, W, al, precalc):
        '''Hessian of Q w.r.t. u = -(alpha I + MT)'''
        U, SigmaVT, M = precalc
        s = u.shape[0]
        A = m*np.exp(U@u)
        T = np.dot(U.T * A, U)   # changes with u
        return -((al)*np.identity(s) + M@T)   ###### sign thing
    def get_step_size(u, du, precalc):
        U, SigmaVT, M = precalc
        A = m*np.exp(U@u)
        T = np.dot(U.T * A, U)
        return np.dot(np.dot(du, T), du)

    ### Setup
    if precalc is None:
        svd_threshold = 1e-12   # consider singular values less than threshold 0
        V, Sigma, U = np.linalg.svd(K, False)
        mask = (Sigma/Sigma.max() >= svd_threshold) # drop singular values less than threshold
        U = U.T[:, mask]
        SigmaVT = (V[:, mask] * Sigma[mask]).T
        M = np.dot(SigmaVT * W, SigmaVT.T)
        precalc = (U, SigmaVT, M)
    else:
        U, SigmaVT, M = precalc

    s = M.shape[0]
    u = u_init if u_init is not None else np.zeros(s)
    mu = al
    Q_old = Q_u(u, G, K, m, W, al, precalc, return_all=False)

    ### Search
    import time
    small_dQ = 0
    for i in range(max_iter):
        grad = grad_Q(u, G, K, m, W, al, precalc)
        hess = hess_Q(u, G, K, m, W, al, precalc)
        
        du = np.linalg.solve(hess-mu*np.identity(s), -grad)  # This is what's taking so long on my mac I guess
        step_size = get_step_size(u, du, precalc)
        
        Q_new = Q_u(u+du, G, K, m, W, al, precalc, return_all=False)
        Q_ratio = Q_new/Q_old
        if step_size < step_max_accept and Q_ratio < 1000:
            # Accept step
            u += du
            Q_old = Q_new
            if np.abs(Q_ratio - 1.0) < dQ_threshold:
                small_dQ += 1
                if small_dQ == max_small_dQ:
                    break
            else:
                small_dQ = 0
            if step_size < step_drop_mu:
                mu = mu/mu_multiplier if mu > mu_min else 0.0
        else:
            # Reject step, increase mu
            mu = np.clip(mu*mu_multiplier, mu_min, mu_max)
        
        if inspect:
            format_string = "{:<20}{:<20}{:<20}{:<20}"
            if i==0:
                print(format_string.format(*['Iter', 'Q', 'Step size', 'Grad.']))
                print("-" * 60)
            print(format_string.format(*[i, Q_new, step_size, np.linalg.norm(grad)]))
            
            
    else:
        print(f"Reached max iterations {max_iter} :(")
        
    A = m*np.exp(U@u)
    return A, u

def find_A_cvxpy(G, K, m, W, al, constr_matrix=None, constr_vec=None, inspect=False):
    """Calculate A for given alpha using cvxpy convex optimization package.

    Optimizes Q[A; al] directly over A (rather than reduced space).
    Can optionally include linear symmetry constraints of form B@A=b.

    Args:
        G (array): Imaginary time correlator data (1xL). K (array): Kernel matrix (LxN). m (array): Model function (1xN).
        W (array): ...
        al (float): Fixed alpha value.
        constr_matrix (array, optional): Constraint matrix B (MxN) for linear constraint B*A=b.
        constr_vec (array, optional): Constraint vector b (Mx1) for linear constraint B*A=b.
    Returns:
        A (array): Optimization result, spectral function A(w) (1xN).
    """
    N = K.shape[1]

    # Define variable and objective function Q
    A = cp.Variable(N, pos=True)    
    S = cp.multiply(al, cp.sum(A-m-cp.rel_entr(A, m)))
    chi2 = cp.square(K@(A)-G)@W
    objective = cp.Maximize(S - 0.5*chi2)
    
    # Define constraints (if any)
    constraints = []
    if constr_matrix is not None:
        constraints.append(constr_matrix@A == constr_vec)   # Add linear symmetry constraint
    # Solve problem
    prob = cp.Problem(objective, constraints)
    Q_optimal = prob.solve(verbose=False, warm_start=True)
    if math.isinf(Q_optimal) or math.isnan(Q_optimal):
        print("Invalid optimal objective value. Solution most likely contains negative values near the endpoints.")
    A = A.value
    return A

# ================================= objective funcs  =================================

def Q_u(u, G, K, m, W, al, precalc, return_all=False):
    # precalc not actually necessary but just for simplifying for now
    U, SigmaVT, M = precalc
    A = m*np.exp(U@u)
    S = (A - m - scipy.special.xlogy(A, A/m)).sum()
    KAG = K@A - G
    chi2 = np.dot(KAG*KAG, W)
    if return_all:
        return al*S - 0.5*chi2, S, chi2
    return (al*S - 0.5*chi2)

def Q(A, G, K, m, W, al, return_all=False):
    if np.isnan(A).any():
        # Return NaN of appropriate shape
        if return_all:
            return np.nan, np.nan, np.nan
        return np.nan
    
    S = (A - m - scipy.special.xlogy(A, A/m)).sum()
    KAG = K@A - G
    chi2 = np.dot(KAG*KAG, W)
    if return_all:
        return al*S - 0.5*chi2, S, chi2
    return (al*S - 0.5*chi2)

# ================================= from Edwin's maxent =================================

def gen_grid(nw, x_min, x_max, w_x):
    """
    generate grid with nw points scaled by the function w_x.

    w[i] = w_x((i+0.5)/nw * (x_max-x_min) + x_min)
    dw[i] = w_x((i+1)/nw * (x_max-x_min) + x_min) -
            w_x(i/nw * (x_max-x_min) + x_min)

    returns w, dw
    """
    x_all = np.linspace(x_min, x_max, 2*nw+1)
    w_all = np.apply_along_axis(w_x, 0, x_all)
    return w_all[1::2], np.abs(np.diff(w_all[::2]))

def model_flat(dw):
    return dw/dw.sum()

def kernel_f(beta, tau, w):
    """fermionic kernel: K(tau, w) = exp(-tau*w)/(1+exp(-beta*w))"""
    return np.exp(-tau[:, None]*w)/(1. + np.exp(-beta*w))


def kernel_b(beta, tau, w, sym=True):
    """bosonic kernel: K(tau, w) = w*exp(-tau*w)/(1-exp(-beta*w))"""
    if sym:
        return w*(np.exp(-tau[:, None]*w) + np.exp(-(beta-tau)[:, None]*w)) \
                / (1. - np.exp(-beta*w))
    else:
        return w*np.exp(-tau[:, None]*w)/(1. - np.exp(-beta*w))
