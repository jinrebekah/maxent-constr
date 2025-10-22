import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import scipy
import resource
import pandas as pd

import sys 
import os
if os.path.exists('/oak/stanford/orgs/simes/rebjin/dqmc-dev/util'):
    sys.path.append('/oak/stanford/orgs/simes/rebjin/dqmc-dev/util')
else:
    sys.path.append('/Users/rebekahjin/Documents/devereaux_group/dqmc-dev/util')
import util

from scipy.interpolate import CubicSpline
import math
 
import matplotlib.pyplot as plt
default_figsize = plt.rcParams['figure.figsize']

def maxent(G, K, m, opt_method='Bryan', constr_matrix=None, constr_vec=None, smooth_al=False, al_method='BT', als=np.logspace(8, 1, 1+20*(8-1)), inspect_al=False, inspect_opt=False):
    """MaxEnt method to calculate A(w) for G(tau)=K(tau, w)*A(w) by maximizing Q[A(w); al]=al*S-0.5*chi^2.

    Args:
        G (array): Imaginary time correlator data (1xL). K (array): Kernel matrix (LxN). m (array): Model function (1xN).
        opt_method (str): Optimization method used to maximize Q. Options are:
            - 'Bryan': Bryan's method (only unconstrained).
            - 'cvxpy': Use cvxpy solver (constrained if constr_matrix and constr_vec provided, otherwise unconstrained).
        al_method (str): al selection method. Options are:
            - 'classic': 
            - 'historic':
            - 'Bryan':
            - 'BT':
        constr_matrix (array, optional): Constraint matrix B (MxN) for linear constraints B*A=b (Default: None).
        constr_vec (array, optional): Constraint vector b (Mx1) for linear constraints B*A=b (Default: None).
        als (array): Array of alpha values used in optimal alpha selection.
    Returns:
        A (array): Spectral function A(w) (1xN).
    """
    N = K.shape[1]
    L = K.shape[0]
    nbin = G.shape[0]
    Gavg = G.mean(0)

    # ---------- Change to basis in which C is diagonal to remove correlations between tau ----------
    # sigma, Uc = np.linalg.eigh(np.cov(G.T) / nbin) # COB matrix
    # Uc = Uc.T
    # W = 1.0/sigma
    sigma, Uc = np.linalg.svd(G - Gavg, False)[1:]
    W = (nbin*(nbin-1)) / (sigma * sigma) # equivalent to above, just using svd for some reason
    # Recommended step: check C eigenvalues, too small (W too big) "might make optimizing Q[A] difficult"
    # I think this is an indication there's not enough bins
    W_ratio_max = 1e8
    W_cap = W_ratio_max*W.min()
    n_large = np.sum(W.max() > W_cap)
    if W.max() > W_cap:
        print(f"clipping {n_large} W values to W.min()*{W_ratio_max}")
        W[W > W_cap] = W_cap # Set values of W above W_cap to W_cap
    K = np.dot(Uc, K)
    G = np.dot(Uc, Gavg) # just calling these G and K
    
    # ------------------------------ Calculate Q, chi2, S, lnP, dlnP for all al in als ------------------------------
    tol=1e-7
    
    N = K.shape[1]
    As = np.zeros((als.shape[0], N))
    Qs = np.zeros_like(als)
    Ss = np.zeros_like(als)
    chi2s = np.zeros_like(als)
    lnPs = np.zeros_like(als)
    dlnPs = np.zeros_like(als)
    # statuses = []
    statuses = np.empty(len(als), dtype=object)

    if opt_method == "Bryan":
        # Precalculate SVD matrices
        svd_threshold = 1e-12   # consider singular values less than threshold 0
        V, Sigma, U = np.linalg.svd(K, False)
        mask = (Sigma/Sigma.max() >= svd_threshold) # drop singular values less than threshold
        U = U.T[:, mask]
        SigmaVT = (V[:, mask] * Sigma[mask]).T
        M = np.dot(SigmaVT * W, SigmaVT.T)
        precalc = (U, SigmaVT, M)
        
        s = M.shape[0]
        us = np.zeros((als.shape[0], s))
        for i, al in enumerate(als):
            u_init = us[i-1]
            # config = {'mu_min': al/4.0, 'mu_max': al*1e100, 'mu_init': al}
            As[i], us[i] = find_A_Bryan(G, K, m, W, al, u_init=u_init, precalc=precalc, inspect=inspect_opt)
            Qs[i], Ss[i], chi2s[i], lnPs[i], dlnPs[i] = Q(As[i], G, K, m, W, al)
            statuses[i] = 'optimal'
    elif opt_method == "cvxpy": 
        # Calling prob.solve on the same problem is faster than calling find_A_cvxpy, calculation moved here
        A = cp.Variable(N, pos=True)
        alpha = cp.Parameter(nonneg=True)
        S = cp.sum(A-m-cp.rel_entr(A, m))
        chi2 = cp.square(K@(A)-G)@W
        objective = cp.Maximize(alpha*S - 0.5*chi2)
        constraints = [constr_matrix@A == constr_vec] if constr_matrix is not None else [] # Add linear symmetry constraint
        prob = cp.Problem(objective, constraints)
        for i, al in enumerate(als):
            try:
                alpha.value = al
                Q_optimal = prob.solve(solver=cp.CLARABEL, verbose=False, warm_start=True, tol_feas=tol, tol_infeas_abs=tol, tol_infeas_rel=tol, tol_gap_abs=tol, tol_gap_rel=tol) # Probably more feasibility settings to be adjusted
                # Q_optimal = prob.solve(verbose=True, warm_start=True) # Probably more feasibility settings to be adjusted
                # Q_optimal = prob.solve(solver=cp.SCS, verbose=True, warm_start=True) # Probably more feasibility settings to be adjusted
                As[i] = A.value
                statuses[i] = prob.status
            except Exception as e:
                # print(f"{al:.2e} optimization failed with error: {e}")
                As[i] = np.full(K.shape[1], np.nan) # Make array of nans if the optimization fails
                statuses[i] = 'fail'
            Qs[i], Ss[i], chi2s[i], lnPs[i], dlnPs[i] = Q(As[i], G, K, m, W, al) # nan too if A has nan

    # Filter out nans
    # mask = ~np.isnan(chi2s)
    # als = als[mask]
    # Qs = Qs[mask]
    # Ss = Ss[mask]
    # chi2s = chi2s[mask]
    # lnPs = lnPs[mask]
    # dlnPs = dlnPs[mask]
    # statuses = statuses[mask]

    # ------------------------------ Select optimal al ------------------------------
    optimal_al, A = select_al(als, As, Qs, Ss, chi2s, lnPs, dlnPs, statuses, al_method=al_method, smooth=smooth_al, inspect_al=inspect_al)

    # ------------------------------ Calculate A with optimal al ------------------------------
    if A is None:
        # recalculate A, pick up optimization setup from earlier
        if opt_method == 'Bryan':
            A, _ = find_A_Bryan(G, K, m, W, optimal_al, u_init=u_init, precalc=precalc, inspect=inspect_opt)
        elif opt_method == 'cvxpy':
            alpha.value = optimal_al
            Q_optimal = prob.solve(solver=cp.CLARABEL, verbose=False, warm_start=True, tol_feas=tol, tol_infeas_abs=tol, tol_infeas_rel=tol, tol_gap_abs=tol, tol_gap_rel=tol)
            A = A.value
    
    return A, optimal_al, As, chi2s, lnPs
    
def select_al(als, As, Qs, Ss, chi2s, lnPs, dlnPs, statuses, al_method='BT', smooth=False, inspect_al=False):
    """Selects optimal alpha. 

    Args:
        al_method (str): al selection method. Options are:
            - 'historic'
            - 'classic' 
            - 'Bryan'
            - 'BT'
    Returns:
        al (float): Optimal alpha value.
    """
    mask = ~np.isnan(chi2s)
    
    als = als[mask]
    Qs = Qs[mask]
    Ss = Ss[mask]
    chi2s = chi2s[mask]
    lnPs = lnPs[mask]
    dlnPs = dlnPs[mask]
    statuses = statuses[mask]
    order = als.argsort()
    
    if smooth:
        # Smooth modified BT, currently for use with noisy constrained xy chi2 data
        chi2_fit = scipy.interpolate.make_smoothing_spline(np.log(als[order]), np.log(chi2s[order]), lam=3)
    else:
        # Default BT
        # fit = CubicSpline(np.log(als[order]), np.log(chi2s[order]))
        chi2_fit = scipy.interpolate.make_smoothing_spline(np.log(als[order]), np.log(chi2s[order]), lam=1)
        
    if al_method == 'historic':
        pass
    elif al_method == 'classic':
        # Optimal alpha maximizes posterior probability P(alpha)
        fit = CubicSpline(np.log(als[order]), dlnPs[order])
        roots = fit.roots(extrapolate=False)
        al = np.exp(fit.roots(extrapolate=False)[0])
        A = None
    elif al_method == 'Bryan':
        weights = np.exp(lnPs) # is this right (actually proportional to P(alpha)?)
        Z = np.trapz(weights, x=als) # P normalization
        A = np.trapz(weights[:, None] * As, x=als, axis=0) / Z
        al = None
    elif al_method == 'BT':
        # Select optimal alpha based on curvature of log-log plot of chi2 vs. al
        k = chi2_fit(np.log(als), 2)/(1 + chi2_fit(np.log(als), 1)**2)**8
        # k = chi2_fit(np.log(als), 2)/(1 + chi2_fit(np.log(als), 1)**2)**1.5
        # gamma = 0.5
        # chi2_fit_BT = scipy.interpolate.make_smoothing_spline(gamma*np.log(als[order]), np.log(chi2s[order]), lam=1)
        # k = chi2_fit_BT(np.log(als), 2)
        al_idx = k.argmax()
        al = als[al_idx]
        A = As[al_idx]
    else:
        raise ValueError(f"Unknown al_method '{al_method}'. Must be one of: 'historic', 'classic', 'Bryan', 'BT'.")

    ### Optional plots for debugging
    if inspect_al:
        # Report how many failed to solve
        print(f"Als failed to solve: {(~mask).sum()}")
        # print(als[~mask])
        if al is not None:
            print(f"Optimal chi2: {np.exp(chi2_fit(np.log(al)))}")

        # Plot chi2 vs. al showing al selection and spline fit, with second derivative peaks.
        # Also plot whether points were 'optimal_inaccurate'
        fig, ax = plt.subplots(ncols=1, figsize=(default_figsize[0], default_figsize[1]))
        # ax.loglog(als, np.exp(fit(np.log(als))), color='r', label='f', zorder=-5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.scatter(als[statuses=='optimal'], chi2s[statuses=='optimal'], s=1.5)
        ax.scatter(als[statuses=='optimal_inaccurate'], chi2s[statuses=='optimal_inaccurate'], s=3)
        
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$\chi^2$")
        if al is not None:
            ax.axvline(al, color='g', label = rf"$\alpha$ = {np.round(al, 2)}")
            ax.annotate(rf"$\alpha$ = {np.round(al, 2)}", (0.05, 0.9), xycoords='axes fraction', fontsize=10, color='g')
        ax2 = ax.twinx()
        ax2.plot(als, np.exp(lnPs), 'g.', ms=3)
        # ax2.plot([al, al], [0, np.exp(lnPs.max())], 'g', lw=1)
        ax2.set_ylabel(r"$P(\alpha)$")

        if al_method == 'BT':
            ax.loglog(als, np.exp(chi2_fit(np.log(als))), color='r', label='f', zorder=-5)
            fig, ax = plt.subplots()
            ax.plot(als, k)
            ax.set_xscale("log")
            ax.set_ylabel(r"$f''/(1 + f'^2)^{1.5}$")
        # ax[1].plot(als, fit(np.log(als), 2)) # Plot 2nd derivative directly
        # ax[1].plot(als, fit(np.log(als), 1)) # Also plot 1st derivative
        # if smooth:
        # ax[1].scatter(als[peaks], k[peaks], s=5)
        # ax[1].scatter(als[al_idx], k[al_idx], color='g', s=5)

        # Plot constraint residuals
        # if smooth:
        #     fig, ax = plt.subplots()
        #     resids = [np.abs(constr_matrix@A-constr_vec) for A in As]   # each resid is vec length N/2
        #     resids_max = [max(resid) for resid in resids]
        #     ax.scatter(als, resids_max)
        #     ax.set_xscale('log')
        
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
            if al is not None:
                ax[i].axvline(al, color='g')
        if al is not None:
            ax[0].annotate(rf"$\alpha$ = {np.round(al, 2)}", (0.05, 0.9), xycoords='axes fraction', fontsize=10, color='g')
        ax[0].set_yscale("log")
        plt.show()
    return al, A

def find_A_Bryan(G, K, m, W, al, u_init=None, precalc=None, inspect=False):
    """Calculate A for given alpha using Bryan's optimization algorithm.

    Bryan's algorithm optimizes Q over a smaller singular space using unconstrained Newton's method (with Marquardt-Levenberg),
    finding optimal u* where A*=m exp(Uu*). Adapted directly from dqmc-dev/util/maxent.py.

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
    # max_small_dQ = 10  # stop if dQ/Q < dQ_threshold this many times in a row
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
        T = np.dot(U.T * A, U)
        return -((al)*np.identity(s) + M@T)
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
    u = u_init if u_init is not None else np.zeros(s, dtype=G.dtype)
    mu = al
    Q_old, *_ = Q_u(u, G, K, m, W, al, precalc)

    ### Search
    small_dQ = 0
    for i in range(max_iter):
        grad = grad_Q(u, G, K, m, W, al, precalc)
        hess = hess_Q(u, G, K, m, W, al, precalc)
        
        du = np.linalg.solve(hess-mu*np.identity(s), -grad)
        step_size = get_step_size(u, du, precalc)
        
        Q_new, *_ = Q_u(u+du, G, K, m, W, al, precalc)
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
            # Supposed to print a table of values for inspection but doesn't look very good lmao
            format_string = "{:<20}{:<20}{:<20}{:<20}"
            if i==0:
                print(format_string.format(*['Iter', 'Q', 'Step size', 'Grad.']))
                print("-" * 60)
            print(format_string.format(*[i, Q_new, step_size, np.linalg.norm(grad)]))   
    else:
        print(f"Reached max iterations {max_iter} :(")
        
    A = m*np.exp(U@u)
    return A, u

# def find_A_cvxpy(G, K, m, W, al, A_init=None, constr_matrix=None, constr_vec=None, inspect=False, tol=1e-8):
#     # Don't use this, it takes too long
#     """Calculate A for given alpha using cvxpy convex optimization package.

#     Optimizes Q[A; al] directly over A (rather than reduced space).
#     Can optionally include linear symmetry constraints of form B@A=b.

#     Args:
#         G (array): Imaginary time correlator data (1xL). K (array): Kernel matrix (LxN). m (array): Model function (1xN).
#         W (array): ...
#         al (float): Fixed alpha value.
#         constr_matrix (array, optional): Constraint matrix B (MxN) for linear constraint B*A=b.
#         constr_vec (array, optional): Constraint vector b (Mx1) for linear constraint B*A=b.
#     Returns:
#         A (array): Optimization result, spectral function A(w) (1xN).
#     """
#     N = K.shape[1]

#     # Define variable and objective function Q
#     A = cp.Variable(N, pos=True)    
#     S = cp.multiply(al, cp.sum(A-m-cp.rel_entr(A, m)))
#     chi2 = cp.square(K@(A)-G)@W
#     objective = cp.Maximize(S - 0.5*chi2)

#     # Define constraints (if any)
#     constraints = []
#     if constr_matrix is not None:
#         constraints.append(constr_matrix@A == constr_vec)   # Add linear symmetry constraint
#     # Solve problem
#     prob = cp.Problem(objective, constraints)
#     if np.any(A_init):
#         A.value = A_init
#     Q_optimal = prob.solve(solver=cp.CLARABEL, verbose=False, warm_start=True, tol_feas=tol, tol_infeas_abs=tol, tol_infeas_rel=tol, tol_gap_abs=tol, tol_gap_rel=tol)
#     if math.isinf(Q_optimal) or math.isnan(Q_optimal):
#         print("Invalid optimal objective value. Solution most likely contains negative values near the endpoints.")
#     A = A.value
#     return A

# ================================= objective funcs  =================================

def Q_u(u, G, K, m, W, al, precalc):
    # precalc not actually necessary but just for simplifying for now
    U, SigmaVT, M = precalc
    A = m*np.exp(U@u)
    S = (A - m - scipy.special.xlogy(A, A/m)).sum()
    KAG = K@A - G
    chi2 = np.dot(KAG*KAG, W)
    return al*S - 0.5*chi2, S, chi2

def Q(A, G, K, m, W, al):
    if np.isnan(A).any():
        return np.full(5, np.nan)
    
    S = (A - m - scipy.special.xlogy(A, A/m)).sum()
    KAG = K@A - G
    chi2 = np.dot(KAG*KAG, W)
    Q = al*S - 0.5*chi2

    Z = np.sqrt(W[:, None])*K*np.sqrt(A)
    lam = np.linalg.svd(Z, False)[1]**2
    lnP = 0.5*np.log(al/(al + lam)).sum() + Q
    dlnP = np.sum(lam/(al + lam)) / (2*al) + (A - m - scipy.special.xlogy(A, A/m)).sum()
    
    return Q, S, chi2, lnP, dlnP

# ================================= various random deubgging funcs  =================================
def plot_G_tau(G, taus, ax=None, all_bins=False, ylabel=r'$G(\tau)$', title='', label='', color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0]):
    """Plots G(tau). Set all_bins=True to see all bins plotted on top of each other."""
    # first index of G is bin
    taus = taus[:G.shape[1]]

    if ax is None:
        fig, ax = plt.subplots(figsize=(default_figsize[0], default_figsize[1]))

    if all_bins:
        for i, G_bin in enumerate(G):
            ax.plot(taus, G_bin, color = color, label=label if i==0 else None)
    else:
        Gavg = np.mean(G, axis=0)
        ax.plot(taus, Gavg, label=label, color=color)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(r'$\tau$')
    ax.set_title(title)

def check_G_tau_gaussian(G, taus, check_tau, ax=None, xlabel='', title='', label='', color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0]):
    # tau
    # plot histogram of G(tau=check_tau) to see if it's Gaussian (can't imagine it wouldn't be lmao but worth a check)
    taus = taus[:G.shape[1]]
    n_bin = G.shape[0]
    print("Nbin: ", n_bin)

    closest_idx = (np.abs(taus - check_tau)).argmin()
    Gs = G[:, closest_idx]
    print(closest_idx)
    if ax is None:
        fig, ax = plt.subplots(figsize=(default_figsize[0], default_figsize[1]))
    ax.hist(Gs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    
    # calculate skew and kurtosis for all imaginary time
    skews = [scipy.stats.skew(G[:, i]) for i in range(G.shape[1])]
    kurtosis = [scipy.stats.kurtosis(G[:, i]) for i in range(G.shape[1])]
    return skews, kurtosis
    
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

# ================================= other helper funcs =================================

def find_data_folder(dir, nflux, n, U, beta):
    """Find data dir with given params in dir (e.g. 8x8_tp0)"""
    for path, dirnames, filenames in os.walk(dir):
        pattern = r"nflux(\d+)/n([\d.]+)/beta([\d.]+)_U(\d+)"
        # print(path, dirnames, filenames)
        match = re.search(pattern, path)
        
        if match:
            nflux_match = int(match.group(1))
            n_match = float(match.group(2))
            beta_match = float(match.group(3))
            U_match = int(match.group(4))
            if nflux==nflux_match and n==n_match and beta==beta_match and U==U_match:
                return path + '/'
        else:
            continue