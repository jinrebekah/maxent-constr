import numpy as np
import matplotlib.pyplot as plt
import scipy
import glob

import my_my_maxent as maxent
import importlib
importlib.reload(maxent)
import sys
import os
if os.path.exists('/oak/stanford/orgs/simes/rebjin/dqmc-dev/util'):
    sys.path.append('/oak/stanford/orgs/simes/rebjin/dqmc-dev/util')
else:
    sys.path.append('/Users/rebekahjin/Documents/Devereaux Group/dqmc-dev/util')
import util
import jqjq

from tqdm import tqdm
import math
import pandas as pd
import seaborn as sns
import pickle
import re

from pathlib import Path

from scipy.interpolate import CubicSpline
default_figsize = plt.rcParams['figure.figsize']

class sigma:
    def __init__(self, path=None, sigma_type=None, ws=None, dws=None, bs=0, settings_xx={}, settings_xy={}):
        # Store simulation parameters
        self.path = path
        self.U, self.Ny, self.Nx, self.beta, self.L, self.tp = util.load_firstfile(
            path, "metadata/U", "metadata/Nx", "metadata/Ny", "metadata/beta", "params/L", "metadata/t'"
        )
        self.T = 1/self.beta
        self.taus = np.linspace(0, self.beta, self.L + 1)
        # get n from path
        
        pattern = r"nflux(\d+)/n([\d.]+)"
        match = re.search(pattern, path)
        if match:
            self.nflux = int(match.group(1))
            self.n = float(match.group(2))

        self.ws = ws
        self.dws = dws
        self.N = len(ws)
        self.bs = bs
        self.sigma_type = sigma_type

        self.jjq0, self.sign, self.n_sample, self.n_bin = self._load_data(path) # note: sign and jj are already divided by n_sample
        self.chi_xx, self.chi_xy = self._prep_chi()

        # Set solver settings (stupid)
        settings_xx_default = {
            'mdl': 'flat', 
            'krnl': 'symm', 
            'opt_method': 'Bryan',
            'al_method': 'BT',
            'inspect_al': False,
            'smooth_al': False
        }
        self.settings_xx = {**settings_xx_default, **settings_xx}
        self.input_xx = self._get_input(self.settings_xx)

        settings_xy_default = {
            'mdl': 'flat',
            'opt_method': 'Bryan',
            'al_method': 'BT',
            'inspect_al': False,
            'smooth_al': False   # whether to use smoothed alpha selection (necessary for constr. opt_method == 'cvxpy')
        }
        self.settings_xy = {**settings_xy_default, **settings_xy}
        self.input_xy = self._get_input(self.settings_xy)

        # Solve for sigma
        if sigma_type == 'xx' or self.nflux==0:
            self.calc_sigma_xx()
        if sigma_type == 'xy':
            self.calc_sigma_xy()

    def _load_data(self, path):
        """Loads j-j data."""
        # Load in all measurements
        n_samples, sign = util.load(
            path, "meas_uneqlt/n_sample", "meas_uneqlt/sign"
        )
        
        # Only keep measurements from bins where n_sample is max value (i.e., bin is complete)
        n_sample = n_samples.max()
        mask = n_samples == n_sample
        sign = sign[mask]
        n_bin = mask.sum()

        # Get jjq0 (can do nonzero t')
        jjq0 = jqjq.get_component(path, 'jj')
        
        # I guess also note that jj does not include tau=beta, remember len(taus) != L
        return jjq0/n_sample, sign/n_sample, n_sample, n_bin

    def _prep_chi(self, symm=True):
        """Gets averaged chi_xx and chi_xy correlators."""
        jj_xx, jj_yy, jj_xy, jj_yx = jqjq.electrical_sum(self.path, self.jjq0) # already divided by n_sample
        
        # Get average longitudinal jj
        chi_xx = 0.5 * (-jj_xx - jj_yy) # average over xx and yy to get avg longitudinal j-j
        if symm:
            chi_xx = 0.5 * (chi_xx + chi_xx[:, -np.arange(self.L) % self.L]) # symmetrize bin by bin
        chi_xx = np.real(chi_xx) # added for nflux != 0 data, should be purely real
        print(chi_xx)
        # Get average transverse jj
        chi_xy = 0.5*(-jj_xy+jj_yx)
        if symm:
            chi_xy = np.concatenate((np.expand_dims(chi_xy[:, 0], axis=1), 0.5*(chi_xy[:, 1:] - chi_xy[:, :0:-1])), axis=1) # stupid but antisymmetrize bin by bin
        chi_xy = 1j*np.imag(chi_xy) # added for nflux != 0 data, should be purely imaginary
        return chi_xx, chi_xy
    
    def _get_input(self, settings):
        """Kinda dumb but this generates a dict with values corresponding to settings dict."""
        # Returns input dict, which are parameters directly passed to MaxEnt
        # mdl = maxent.model_flat(self.dws) if settings['mdl'] == 'flat' else settings['mdl']
        if isinstance(settings['mdl'], str) and settings['mdl'] == 'flat':
            mdl = maxent.model_flat(self.dws)
        else:
            mdl = settings['mdl']
        if 'krnl' in settings and settings['krnl'] == 'symm':
            krnl = maxent.kernel_b(self.beta, self.taus[0 : self.L // 2 + 1], self.ws[self.N//2:], sym=True)
            mdl = mdl[self.N//2:]
        else:
            krnl = maxent.kernel_b(self.beta, self.taus[:-1], self.ws, sym=False)
        # make sure mdl is normalized
        mdl = mdl/np.sum(mdl)
        opt_method = settings['opt_method']
        al_method = settings['al_method']
        smooth_al = settings['smooth_al'] if 'smooth_al' in settings else False
        # als = np.logspace(8, 1, 1+20*(8-1)) if 'krnl' in settings else np.logspace(8, 2, 1+20*(8-1))
        als = np.logspace(8, 1, 1+20*(8-1)) if 'krnl' in settings else np.logspace(8, 3, 1+20*(8-3))
        
        return {'m': mdl, 'K': krnl, 'opt_method': opt_method, 'al_method': al_method, 'smooth_al': smooth_al, 'als': als}

    def calc_sigma_xx(self):
        if self.bs:
            bs_list = []
            for i in tqdm(range(self.bs), desc='Sigma_xx bootstraps'):
                resample = np.random.randint(0, self.n_bin, self.n_bin)
                re_sigmas_xx, debug_vals = self._calc_sigma_xx_bins(resample, return_As=False)
                bs_dict = {'re_sig_xx': re_sigmas_xx, 'resample': resample, **debug_vals}
                bs_list.append(bs_dict)
        else:
            all_bins = np.arange(self.n_bin)
            re_sigmas_xx, debug_vals = self._calc_sigma_xx_bins(all_bins, inspect_al = self.settings_xx['inspect_al'], return_As=False)
            bs_list = [{'re_sig_xx': re_sigmas_xx, 'resample': all_bins, **debug_vals}]
        # Create results dataframe from list of bs dicts
        self.results = pd.DataFrame(bs_list)
    
    def _calc_sigma_xx_bins(self, resample, inspect_al=False, return_As=False):
        """Calculates sigma_xx for bin indices specified by resample."""
        
        f = self.chi_xx[resample].mean(0)
        chiq0w0 = CubicSpline(self.taus, np.append(f, f[0])).integrate(0, self.beta)
        if self.settings_xx['krnl'] == 'symm':
            # Symmetric krnl, with half tau and w range. Only unconstrained option
            g = self.chi_xx[resample, : self.L // 2 + 1] / (chiq0w0/2) # when we truncate taus, it includes the midpoint. include factor of 2 for norm.
            A_xx, al_xx, As_xx, chi2s_xx, lnPs_xx = maxent.maxent(g, **self.input_xx, inspect_al=inspect_al)
            A_xx = np.concatenate((A_xx[::-1], A_xx))/2 # Fill in the negative w half of A_xx and remove factor of 2. A_xx now properly normalized to 1
            As_xx = np.concatenate((As_xx[:, ::-1], As_xx), axis=1)/2

            # g = self.chi_xx[resample, : self.L // 2 + 1] / (chiq0w0)
            # A_xx, al_xx, As_xx, chi2s_xx, lnPs_xx = maxent.maxent(g, **self.input_xx, inspect_al=inspect_al)
            # A_xx = np.concatenate((A_xx[::-1], A_xx))
        else:
            # Full krnl
            g = self.chi_xx[resample] / chiq0w0
            if self.input_xx['opt_method'] == 'Bryan':
                # Unconstrained
                A_xx, al_xx, As_xx, chi2s_xx, lnPs_xx = maxent.maxent(g, **self.input_xx, inspect_al=inspect_al)
            else:
                # Define symmetry constraint matrices for A_xx
                b = np.zeros(self.N//2)
                B = np.hstack((np.flip(np.identity(self.N//2), axis=0), -1*np.identity(self.N//2)))
                self.input_xx['constr_matrix'] = B
                self.input_xx['constr_vec'] = b
                A_xx, al_xx, As_xx, chi2s_xx, lnPs_xx = maxent.maxent(g, **self.input_xx, inspect_al=inspect_al)
        re_sigmas_xx = np.real(A_xx / self.dws * (chiq0w0 / self.sign[resample].mean()) * np.pi)
        # re_sigmas_xx = np.real(A_xx / self.dws * (chiq0w0 / self.sign.mean()) * np.pi)
        debug_vals = {'A_xx': A_xx, 'norm_xx': chiq0w0, 'al_xx': al_xx, **({'As_xx': As_xx, 'chi2s_xx': chi2s_xx, 'lnPs_xx': lnPs_xx} if return_As else {})} # leave off As_xx, chi2s_xx, lnPs_ss by default
        return re_sigmas_xx, debug_vals

    def calc_sigma_xy(self):
        self.xs = np.linspace(-np.max(self.ws), np.max(self.ws), 1500) # ws used in Kramer's Kronig transform, change to make specifiable later
        if self.bs:
            bs_list = []
            for i in tqdm(range(self.bs), desc='Sigma_xy bootstraps'):
                resample = np.random.randint(0, self.n_bin, self.n_bin)
                re_sigmas_xy, im_sigmas_xy, sigmas_sum, re_sigmas_xx, debug_vals = self._calc_sigma_xy_bins(resample, return_As=False)
                bs_dict = {'re_sig_xx': re_sigmas_xx, 'im_sig_xy': im_sigmas_xy, 'sig_sum': sigmas_sum, 're_sig_xy': re_sigmas_xy, 'resample': resample, **debug_vals}
                bs_list.append(bs_dict)
        else:
            all_bins = np.arange(self.n_bin)
            re_sigmas_xy, im_sigmas_xy, sigmas_sum, re_sigmas_xx, debug_vals = self._calc_sigma_xy_bins(all_bins, inspect_al = self.settings_xy['inspect_al'], return_As=False)
            bs_list = [{'re_sig_xx': re_sigmas_xx, 'im_sig_xy': im_sigmas_xy, 'sig_sum': sigmas_sum, 're_sig_xy': re_sigmas_xy, 'resample': all_bins, **debug_vals}]
        # Create results dataframe from list of bs dicts
        self.results = pd.DataFrame(bs_list)

    def _calc_sigma_xy_bins(self, resample, inspect_al=False, return_As=False):
        """Calculates sigma_xy for bin indices specified by resample."""
        # Get sigma_xx
        re_sigmas_xx, debug_vals_xx = self._calc_sigma_xx_bins(resample, return_As=return_As)
        A_xx = debug_vals_xx['A_xx']
        # Maxent sum
        f = np.append(self.chi_xx[resample].mean(0), self.chi_xx[resample].mean(0)[0]) - np.real(1j*np.append(self.chi_xy[resample].mean(0), -self.chi_xy[resample].mean(0)[0]))
        chiq0w0 = CubicSpline(self.taus, f).integrate(0, self.beta)
        g = (self.chi_xx[resample] - np.real(1j*self.chi_xy[resample])) / chiq0w0
        if self.input_xy['opt_method'] == 'Bryan':
            # Unconstrained
            A_sum, al_sum, As_sum, chi2s_sum, lnPs_sum = maxent.maxent(g, **self.input_xy, inspect_al = inspect_al)
        elif self.input_xy['opt_method'] == 'cvxpy':
            # Define symmetry constraint matrices
            b = 2*A_xx[self.N//2:]
            B = np.hstack((np.flip(np.identity(self.N//2), axis=0), np.identity(self.N//2)))
            self.input_xy['constr_matrix'] = B
            self.input_xy['constr_vec'] = b
            A_sum, al_sum, As_sum, chi2s_sum, lnPs_sum = maxent.maxent(g, **self.input_xy, inspect_al = inspect_al)
        sigmas_sum = np.real(A_sum / self.dws * (chiq0w0 / self.sign[resample].mean())) * np.pi
        # np.real(A_xx / self.dws * (chiq0w0 / self.sign[resample].mean()) * np.pi)
        im_sigmas_xy = sigmas_sum-re_sigmas_xx
        # Kramer's Kronig for re_sigma_xy
        ys = CubicSpline(self.ws, im_sigmas_xy)(self.xs)
        re_sigmas_xy = -np.imag(scipy.signal.hilbert(ys))

        debug_vals = {'norm_sum': chiq0w0, 'A_sum': A_sum, 'A_xy': A_sum-A_xx, 'al_sum': al_sum, **({'As_sum': As_sum, 'chi2s_sum': chi2s_sum, 'lnPs_sum': lnPs_sum} if return_As else {}), **debug_vals_xx}
        return re_sigmas_xy, im_sigmas_xy, sigmas_sum, re_sigmas_xx, debug_vals

    def print_summary(self):
        # Print summary of settings used in opt
        pass

    def get_chi_xx(self, bs=None, include_beta=True):
        '''Reproduces G_xx(tau) for specified bootstrap. If no bootstrap, just compares means'''
        if bs is None:
            A_xx = (self.results['A_xx']*self.results['norm_xx']).mean()
            chi_xx = np.mean(self.chi_xx, axis=0)
        else:
            resample = self.results['resample'][bs]
            A_xx = self.results['A_xx'][bs]*self.results['norm_xx'][bs]
            chi_xx = np.mean(self.chi_xx[resample], axis=0)
            
        if self.settings_xx['krnl']=='symm':
            # A_xx full length, but krnl is not 
            KA = self.input_xx['K']@A_xx[self.N//2:]   # only for the first half of taus
            KA = np.concatenate((KA, KA[math.ceil(self.L/2)-1::-1]))[:-1] # without including beta point
        else:
            KA = self.input_xx['K']@A_xx
            
        if include_beta:
            KA = np.append(KA, KA[0])
            chi_xx = np.append(chi_xx, chi_xx[0])
        
        return KA, chi_xx
    
    def get_chi_xy(self, bs=None, include_beta=True):
        '''Reproduces G_xy(tau)'''
        # Get from df
        # If bs, just average A_xy*norm for all bs
        if bs is None:
            A_xy = (self.results['A_xy']*self.results['norm_sum']).mean()
            chi_xy = np.mean(self.chi_xy, axis=0)
        else:
            resample = self.results['resample'][bs]
            A_xy = self.results['A_xy'][bs]*self.results['norm_sum'][bs]
            chi_xy = np.mean(self.chi_xy[resample], axis=0)
        
        KA = self.input_xy['K']@A_xy
        
        if include_beta:    
            chi_xy = np.append(chi_xy, -chi_xy[0])
            KA = np.append(KA, -KA[0])
        return KA, chi_xy

############################ Calculate stuff ################################
def calc_rho_xx_0(sig):
    # Return DC resistivity + error for sigma object
    
    # Wait jk we prob want a 2D array for re_sig_xx and re_sig_xy, first index is bootstrap
    re_sig_xx_bs = np.array(sig.results['re_sig_xx'].tolist())
    re_sig_xy_bs = np.array(sig.results['re_sig_xy'].tolist())

    nflux = sig.nflux

    sig_xx_0_bs = np.array([scipy.interpolate.CubicSpline(sig.ws, re_sig_xx)(0) for re_sig_xx in re_sig_xx_bs]) # DC xx conductivity for each bootstrap
    # Also modified bc the sig_xy data for nflux=0 is false signal and can't be trusted
    # sig_xy_0_bs = np.zeros_like(sig_xx_0_bs)
    if nflux==0:
        sig_xy_0_bs = np.zeros_like(sig_xx_0_bs)
    else:
        sig_xy_0_bs = np.array([scipy.interpolate.CubicSpline(sig.xs, re_sig_xy)(0) for re_sig_xy in re_sig_xy_bs]) # xy
    
    rho_xx_0_bs = sig_xx_0_bs/(sig_xx_0_bs**2 + sig_xy_0_bs**2)
    # print(rho_xx_0_bs, np.shape(rho_xx_0_bs))
    rho_xx_0 = np.mean(rho_xx_0_bs)
    rho_xx_err = np.std(rho_xx_0_bs)
    sig_xx_0 = np.mean(sig_xx_0_bs)
    sig_xx_err = np.std(sig_xx_0_bs)
    sig_xy_0 = np.mean(sig_xy_0_bs)
    sig_xy_err = np.std(sig_xy_0_bs)
    return rho_xx_0, rho_xx_err, sig_xx_0, sig_xx_err, sig_xy_0, sig_xy_err

def calc_rho_xx_proxy(sig, bs=200):
    # jj_xx, jj_yy, jj_xy, jj_yx = jqjq.electrical_sum(sig.path, sig.jjq0) # already divided by n_sample
    # colors = sns.color_palette('husl', 2)
    # for bin in range(sig.n_bin):
    #     plt.scatter(sig.taus[:-1], sig.chi_xx[bin], color=colors[0])
    # dt = sig.beta/sig.L
    proxy1_list = []
    proxy2_list = []
    for i in range(bs):
        resample = np.random.randint(0, sig.n_bin, sig.n_bin)
        # calculate proxy
        chi_fit =  CubicSpline(sig.taus[:-1], np.mean(sig.chi_xx[resample], axis=0))
        chi_half_beta = chi_fit(sig.beta/2)
        dchi_half_beta = chi_fit.derivative(2)(sig.beta/2)
        # print(chi_half_beta)
        proxy1_list.append(np.pi/(sig.beta**2*chi_half_beta))
        proxy2_list.append(dchi_half_beta/(2*np.pi*chi_half_beta**2))
    proxy1 = np.mean(proxy1_list)
    proxy1_err = np.std(proxy1_list)
    proxy2 = np.mean(proxy2_list)
    proxy2_err = np.std(proxy2_list)
    return proxy1, proxy1_err, proxy2, proxy2_err

def calc_sigma_xy_proxy(sig, bs=200):
    # probably missing factors of whatever but proxy should be roughly proportional to sigma_xy lol...

    proxy_list = []
    for i in range(bs):
        resample = np.random.randint(0, sig.n_bin, sig.n_bin)
        chi_fit =  CubicSpline(sig.taus[:-1], np.mean(np.imag(sig.chi_xy)[resample], axis=0)) # this is antisymmetrized

        chi_slope = chi_fit.derivative(1)(sig.beta/2)
        proxy_bs = beta**3*chi_slope
        proxy_list.append(proxy_bs)

    proxy = np.mean(proxy_list)
    proxy_err = np.std(proxy_list)
    return proxy, proxy_err



############################ Various badly written plotting and debugging funcs ################################

def plot_results(sig, sig_names=None, bs_idx=None, bs_mode='errorbar'):
    # Plots sig results. Can give it bs indices to only plot specific bootstraps, otherwise plots all
    if sig_names is None:
        if sig.sigma_type == 'xx':
            sig_names = ['re_sig_xx']
        else:
            sig_names = ['re_sig_xx', 'sig_sum', 'im_sig_xy', 're_sig_xy']

    num_plots = len(sig_names)
    plot_size = plt.rcParams['figure.figsize']
    fig, ax = plt.subplots(ncols=num_plots, figsize=(plot_size[0]*num_plots, plot_size[1]), layout='constrained')

    if num_plots==1:
        plot_sigma(sig, ax, sig_names[0], bs_idx, bs_mode=bs_mode)
    else:
        for i in range(num_plots): plot_sigma(sig, ax[i], sig_names[i], bs_idx=bs_idx, bs_mode=bs_mode)
    
    try:
        fig.suptitle(rf'nflux = {sig.nflux}, n = {sig.n}, U = {sig.U}, $\beta$ = {sig.beta}, bs = {sig.bs}')
    except:
        fig.suptitle(rf'U = {sig.U}, $\beta$ = {sig.beta}, bs = {sig.bs}')
    # plt.tight_layout()
    plt.show()

def plot_sigma(sig, ax, sigma_name, bs_idx=None, bs_mode='errorbar', color='#0C5DA5', label='', show_opt_info=True):
    sigma_name_dict = {
        "re_sig_xx": r'Re[$\sigma_{xx}(\omega)$]', 
        # "im_sig_xx": r'Im[$\sigma_{xx}(\omega)$]',
        "re_sig_xy": r'Re[$\sigma_{xy}(\omega)$]',
        "im_sig_xy": r'Im[$\sigma_{xy}(\omega)$]',
        "sig_sum": r'Re[$\sigma_{xx}(\omega)$] + Im[$\sigma_{xy}(\omega)$]'
    }

    ws = sig.xs if sigma_name == 're_sig_xy' else sig.ws
    
    if sig.bs:
        if bs_idx is None:
            # Plot all bootstraps
            bs_idx = np.arange(sig.bs) # bs_idx needs to be a list
        sig_bs = np.array(sig.results[sigma_name].tolist())[bs_idx] # Only keep bs we want to plot
        if len(sig_bs) == 1:
            bs_mode = 'all'   # no such thing as std for 1 bs, use 'all' mode
        if bs_mode=='errorbar':
            # Plot bootstrap mean with std error bars
            markers, caps, bars = ax.errorbar(ws, np.mean(sig_bs, axis=0), yerr=np.std(sig_bs, axis=0), fmt='s-', lw=0.7, ms=0, capsize=0, color=color, ecolor='orange', elinewidth=0.5, label=label)
            # [bar.set_alpha(0.0001) for bar in bars]
        else:
            # Plot all bootstraps on top of each other
            for i in range(len(sig_bs)):
                ax.plot(ws, sig_bs[i], lw=0.7, color=color, alpha=0.15, label=label if i==0 else None)
    else:
        # Plot all bins result
        ax.plot(ws, sig.results[sigma_name][0], color=color)

    # Annotate with opt info in top left corner I guess
    if show_opt_info:
        settings = sig.settings_xx if 'xx' in sigma_name else sig.settings_xy
        method = settings['opt_method']
        K = sig.settings_xx['krnl']
        al_method = 'smooth' if settings['smooth_al'] else 'default'
        ax.annotate('O: ' + method + '\n' + r'$K_{xx}$: ' + K +'\n'+r'$\alpha$: '+ al_method, (0.04, 0.80), xycoords='axes fraction', fontsize=8, color='gray')
        # ax.annotate(f'O: {opt_method_dict[method]} \n$K_{xx}$: {K}', (0.03, 0.89), xycoords='axes fraction')
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(sigma_name_dict[sigma_name])

    # ax.set_title(rf'U = {sig.U}, $\beta$ = {sig.beta}')

def inspect_al(sig, sig_type, bs, ws_plot=[0], als_plot=None, w_lim=None, redo=False):
    # Jk actually just redo the bootstrap essentially lmfao just to see the alpha selection plot
    # Also include color plot of spectra vs. alpha
    # running calc_sigma_*_bins is very expensive
    resample = sig.results['resample'][bs]
    
    # Ensure cache df exists
    if not hasattr(sig, '_al_cache'):
        print('No cache, recalculating everything.')
        sig._al_cache = {} # dict I guess
    cache = sig._al_cache
    
    if sig_type == 'xx':
        if bs not in cache or redo:
            # compute xx for the first time and store in _al_cache
            re_sigmas_xx, debug_vals = sig._calc_sigma_xx_bins(resample, inspect_al = redo, return_As=True)
            A_xx_vs_al, chi2_xx_vs_al, lnP_xx_vs_al = debug_vals['As_xx'], debug_vals['chi2s_xx'], debug_vals['lnPs_xx']
            norm_xx, al_xx = debug_vals['norm_xx'], debug_vals['al_xx']
            cache[bs] = {'re_sig_xx': re_sigmas_xx, **debug_vals}
        else:
            # xx has been computed before, grab from _al_cache in sig
            print('Grabbing from cache')
            re_sigmas_xx = cache[bs]['re_sig_xx']
            A_xx_vs_al = cache[bs]['As_xx']
            chi2_xx_vs_al = cache[bs]['chi2s_xx']
            lnP_xx_vs_al = cache[bs]['lnPs_xx']
            norm_xx = cache[bs]['norm_xx']
            al_xx = cache[bs]['al_xx']
            
        # need re_sigmas_xx, A_xx_vs_al, chi2_xx_vs_al, norm_xx, al_xx
    elif sig_type == 'xy':     
        if not (bs in cache and 'As_sum' in cache[bs]) or redo:
            # compute xx and xy for the first time and store in _al_cache
            re_sigmas_xy, im_sigmas_xy, sigmas_sum, re_sigmas_xx, debug_vals = sig._calc_sigma_xy_bins(resample, inspect_al = redo, return_As=True)
            A_xx_vs_al, chi2_xx_vs_al, lnP_xx_vs_al = debug_vals['As_xx'], debug_vals['chi2s_xx'], debug_vals['lnPs_xx']
            norm_xx, al_xx = debug_vals['norm_xx'], debug_vals['al_xx']
            A_sum_vs_al, chi2_sum_vs_al, lnP_sum_vs_al = debug_vals['As_sum'], debug_vals['chi2s_sum'], debug_vals['lnPs_sum']
            norm_sum, al_sum = debug_vals['norm_sum'], debug_vals['al_sum']
            cache[bs] = {'re_sig_xx': re_sigmas_xx, 'im_sig_xy': im_sigmas_xy, **debug_vals}
        else:
            print('Grabbing from cache')
            # xy has been computed before, grab from _al_cache in sig
            re_sigmas_xx = cache[bs]['re_sig_xx']
            im_sigmas_xy = cache[bs]['im_sig_xy']
            A_xx_vs_al = cache[bs]['As_xx']
            chi2_xx_vs_al = cache[bs]['chi2s_xx']
            lnP_xx_vs_al = cache[bs]['lnPs_xx']
            norm_xx = cache[bs]['norm_xx']
            al_xx = cache[bs]['al_xx']
            
            A_sum_vs_al = cache[bs]['As_sum']
            chi2_sum_vs_al = cache[bs]['chi2s_sum']
            lnP_sum_vs_al = cache[bs]['lnPs_sum']
            norm_sum = cache[bs]['norm_sum']
            al_sum = cache[bs]['al_sum']
            
    # if sig.settings_xx['krnl'] == 'symm':
    #     A_xx_vs_al = np.concatenate((A_xx_vs_al[:, ::-1], A_xx_vs_al), axis=1)
    sigmas_xx_vs_al = np.real(A_xx_vs_al/sig.dws * (norm_xx/sig.sign[resample].mean())*np.pi)
    
    if sig_type == 'xx':
        # See color plot of sig_xx spectra vs. alphas
        # A_vs_al = A_xx_vs_al
        sigmas_vs_al = sigmas_xx_vs_al
        sigmas = re_sigmas_xx
        optimal_al = al_xx
        chi2s = chi2_xx_vs_al
        lnPs = lnP_xx_vs_al
        sig_label = r'Re[$\sigma_{xx}(\omega)$]'
        als = sig.input_xx['als']
    else: # if sig_type == 'xy'
        # See color plot of im_sig_xy vs. alphas
        # sigmas_xx_al = As_xx/sig.dws * (sig.results['norm_xx'][bs]/sig.sign[resample].mean())*np.pi # not necessary
        sigmas_sum_vs_al = np.real((A_sum_vs_al)/sig.dws * (norm_sum/sig.sign[resample].mean()))*np.pi
        sigmas_vs_al = sigmas_sum_vs_al - re_sigmas_xx
        sigmas = im_sigmas_xy
        optimal_al = al_sum
        chi2s = chi2_sum_vs_al
        lnPs = lnP_sum_vs_al
        sig_label = r'Im[$\sigma_{xy}(\omega)$]'
        als = sig.input_xy['als']

    if np.any(als_plot)==None:
        als_plot=[]
    als_plot = np.array(als_plot)
    # als_plot = np.append(als_plot, optimal_al) # always plot optimal al

    # Plot density plot of sigma vs. al, with neighboring plot of spectra at alpha slices in als_plot
    fig, ax = plt.subplots(figsize = (default_figsize[0]*3, default_figsize[1]), ncols=3, layout='constrained')

    # Chi2 plot
    ax[0].scatter(als, chi2s)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$\alpha$')
    ax[0].set_ylabel(r'$\chi^2$')
    # also should include P(alpha)f
    ax2 = ax[0].twinx()
    ax2.plot(als, np.exp(lnPs), 'g.', ms=3)
    # ax2.plot([al, al], [0, np.exp(lnPs.max())], 'g', lw=1)
    ax2.set_ylabel(r"$P(\alpha)$")

    # Color plot
    lim = max(np.nanmin(sigmas_vs_al), np.nanmax(sigmas_vs_al))
    # print(lim)
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    X, Y = np.meshgrid(als, sig.ws)
    # pcol = ax[1].pcolormesh(X, Y, np.transpose(sigmas_vs_al), cmap='plasma', rasterized=True, norm=norm)
    pcol = ax[1].pcolormesh(X, Y, np.transpose(sigmas_vs_al), cmap='plasma', rasterized=True)
    ax[1].invert_yaxis()
    ax[1].set_xscale('log')
    fig.colorbar(pcol, ax=ax[1])
    ax[1].set_xlabel(r'$\alpha$')
    ax[1].set_ylabel(r'$\omega$')
    ax[1].set_ylim(-20, 20)

    # Spectrum plot
    colors = sns.color_palette('tab10', len(als_plot))
    if optimal_al is not None:
        label = rf'$\alpha$ = {optimal_al: .2e}'
        for j in range(2): ax[j].axvline(optimal_al, color='r') # Plot lines on colorplot and chi2 plots at als_plot
    else:
        label = 'Bryan'
    ax[2].plot(sig.ws, sigmas, color='r', label=label)
    
    for i, al_plot in enumerate(als_plot):
        color = colors[i]
        al_idx = find_nearest(als, al_plot, get_idx=True)
        ax[2].plot(sig.ws, sigmas_vs_al[al_idx], color=color, label=rf'$\alpha$ = {al_plot: .2e}')
        for j in range(2): ax[j].axvline(al_plot, color=color) # Plot lines on colorplot and chi2 plots at als_plot

    ax[2].set_xlabel(r'$\omega$')
    ax[2].set_ylabel(sig_label)
    if np.any(w_lim):
        ax[2].set_xlim(*w_lim)
    else:
        ax[2].set_xlim(-20, 20)
    ax[2].legend()


    ### BT-suggested diagnostic plots
    # another plot of spectrum value at ws in ws_plot vs. al
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    for w_plot in ws_plot:
        idx = np.argmin(np.abs(sig.ws-w_plot))
        sigma_vals = sigmas_vs_al[:, idx]
        ax.plot(als, sigma_vals)
    
    # plot of (xx for now fml) residuals basically vs. tau, for one alpha
    fig, ax = plt.subplots()
    G = np.mean(sig.chi_xx[resample, :] / (sig.results['norm_xx'][bs]), axis=0)
    als_resid = [optimal_al, *als_plot]
    for al_resid in als_resid:
        idx = np.argmin(np.abs(als-al_resid))
        A_xx = A_xx_vs_al[idx]
        if sig.settings_xx['krnl']=='symm':
            # A_xx full length, but krnl is not
            KA = sig.input_xx['K']@A_xx[sig.N//2:]   # only for the first half of taus
            KA = np.concatenate((KA, KA[math.ceil(sig.L/2)-1::-1]))[:-1] # without including beta point
        else:
            KA = sig.input_xx['K']@A_xx 
        deltaG = G-KA
        plt.scatter(sig.taus[:len(deltaG)], deltaG, s=8)
        plt.plot(sig.taus[:len(deltaG)], deltaG, lw=1)
        # plt.scatter(sig.taus[:len(deltaG)], KA)
        # plt.scatter(sig.taus[:len(deltaG)], G)
    ax.axhline(0, lw=1, ls='--', color='gray')
    
    plt.show()
    

def compare_chi_tau(sigs, mode='xx', bs=0):
    """Plots asdf."""
    # Verify that sig1 and sig2 have the same data
    sig1 = sigs[0]
    taus = sig1.taus
    U = sig1.U
    beta = sig1.beta
    bs = sig1.bs
    if mode == 'xx':
        _, chi = sig1.get_chi_xx()
        chi_label = r'$\chi_{xx}(\tau)$'
        KAs = [KA for KA, chi_xx in (sig.get_chi_xx() for sig in sigs)]
        labels = [r'$KA$ Bryan' if sig.settings_xx['opt_method'] == 'Bryan' else r'$KA$ Constr.' for sig in sigs]
    else:
        _, chi = sig1.get_chi_xy()
        chi = np.real(-1j*chi)
        chi_label = r'$-i\chi_{xy}(\tau)$'
        KAs = [KA for KA, chi_xy in (sig.get_chi_xy() for sig in sigs)]
        labels = [r'$KA$ Bryan' if sig.settings_xy['opt_method'] == 'Bryan' else r'$KA$ Constr.' for sig in sigs]
    # resids = [KA-chi for KA in KAs]
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = color_cycle[1:3]

    plot_size = plt.rcParams['figure.figsize']
    fig, ax = plt.subplots(ncols=2, figsize=(plot_size[0]*2, plot_size[1]*1.2), layout='constrained')
    ax[0].plot(taus, chi, label=chi_label)
    for i in range(len(sigs)): ax[0].plot(taus, KAs[i], label=labels[i])
    ax[0].set_title('Data')
    ax[0].legend()

    # for i in range(len(sigs)): ax[1].scatter(taus, resids[i], color=colors[i], s=7)
    ax[1].axhline(0, color='gray', ls='--', alpha=0.5)
    # ax[1].set_ylabel('Residuals')
    ax[1].set_title('Residuals')

    for i in range(2): ax[i].set_xlabel(r'$\tau$')
    fig.suptitle(rf'U = {U}, $\beta$ = {beta}, bs = {bs}')
    # plt.tight_layout()
    plt.show()

def check_KA(sig, mode='xx', bs=None):
    """Plot G(tau) and KA with residuals."""
    # basically rn this is a version of compare_chi_tau that just checks it for specific bs, probably needs to be merged with it at some point
    taus = sig.taus
    if mode=='xx':
        KA, chi = sig.get_chi_xx(bs=bs)
        chi_label = r'$\chi_{xx}(\tau)$'
    else:
        KA, chi = sig.get_chi_xy(bs=bs)
        chi = np.real(-1j*chi)
        chi_label = r'$-i\chi_{xy}(\tau)$'
    resids = KA-chi
    
    plot_size = plt.rcParams['figure.figsize']
    fig, ax = plt.subplots(ncols=2, figsize=(plot_size[0]*2, plot_size[1]*1.2), layout='constrained')
    ax[0].plot(taus, chi, label=chi_label)
    ax[0].plot(taus, KA, label='KA')
    ax[0].set_title('Data')
    ax[0].legend()
    
    ax[1].scatter(taus, resids, s=7)
    ax[1].set_title('Residuals')

def inspect_symm(sig, bs=0):
    # uh plot symmetry residuals of optimal solution for now
    
    im_sig_xy = sig.results['im_sig_xy'].tolist()[bs]

    resids = np.abs(im_sig_xy[sig.N//2:] - (-im_sig_xy[:sig.N//2][::-1]))    # compare right half of re_sig_xy with left half
    
    fig, ax = plt.subplots()
    ax.scatter(sig.ws[sig.N//2:], resids)

def plot_chi_tau(sig, symm=True, all_bins=False):
    """Plots chi_xx and chi_xy."""
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    chi_xx, chi_xy = sig._prep_chi(symm=symm)
    fig, ax = plt.subplots(ncols=2, figsize=(default_figsize[0]*2, default_figsize[1]))
    maxent.plot_G_tau(chi_xx, sig.taus, ax=ax[0], all_bins=all_bins, ylabel=r'$\chi_{xx}(\tau)$')
    maxent.plot_G_tau(np.real(-1j*chi_xy), sig.taus, ax=ax[1], all_bins=all_bins, ylabel=r'$-i\chi_{xy}(\tau)$')
    fig.suptitle(rf'U={sig.U}, $\beta$={sig.beta}, n={sig.n}, nflux={sig.nflux}')
    
    plt.tight_layout()

def check_chi_tau_gaussian(sig, check_tau=None, symm=True, all_bins=False):
    """Plots chi_xx and chi_xy at specified tau to see if Gaussian"""
    chi_xx, chi_xy = sig._prep_chi(symm=symm)
    # chi_xx, chi_xy = sig.chi_xx, sig.chi_xy
    # print(np.all(chi_xx, sig.chi_xx))
    # print(np.all(chi_xy, sig.chi_xy))
    if check_tau is None:
        check_tau = sig.beta/2
    fig, ax = plt.subplots(ncols=2, figsize=(default_figsize[0]*2, default_figsize[1]))
    check_tau_str = rf"$\beta \times ${check_tau/sig.beta:.2f}"
    skews_xx, kurtosis_xx = maxent.check_G_tau_gaussian(chi_xx, sig.taus, check_tau, ax=ax[0], xlabel=rf'$\chi_{{xx}}(\tau = $ {check_tau_str})')
    skews_xy, kurtosis_xy = maxent.check_G_tau_gaussian(np.real(-1j*chi_xy), sig.taus, check_tau, ax=ax[1], xlabel=rf'$-i\chi_{{xy}}(\tau = ${check_tau_str})')
    
    # print df of all skews and kurtoses
    df = pd.DataFrame({
        'tau': sig.taus[:chi_xx.shape[1]],
        'skew_xx': skews_xx,
        'kurtosis_xx': kurtosis_xx,
        'skew_xy': skews_xy,
        'kurtosis_xy': kurtosis_xy
    })
    avg_row = {
    'tau': 'avg',  # Use a label or NaN
    'skew_xx': df['skew_xx'].mean(),
    'kurtosis_xx': df['kurtosis_xx'].mean(),
    'skew_xy': df['skew_xy'].mean(),
    'kurtosis_xy': df['kurtosis_xy'].mean()
    }
    # Append the average row
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    print(df)
    
def get_bs_outliers(sig, mode='xx'):
    """Find bs indices which differ the most from the mean."""
    if mode=='xx':
        sigma_name='re_sig_xx'
    else:
        sigma_name='re_sig_xy'
        
    sig_bs = np.array(sig.results[sigma_name].tolist())
    sig_mean = np.mean(sig_bs, axis=0)
    errs = np.linalg.norm(sig_bs - sig_mean, axis=1)
    
    sorted_indices = np.argsort(errs)[::-1]
    print(sorted_indices)


############################ Loading pickle funcs ################################

def get_sig_pickle(path, nflux=None, n=None, U=None, beta=None):
    # path is either directly to pickle or to folder containing all pickles ('8x8_tp0')
    pickle_path=None
    if path.endswith('.pickle'):
        pickle_path = path
    else:
        pattern = rf"nflux{nflux}/n{n}/beta{beta}_U{U}"
        # pattern = r"nflux(\d+)/n([\d.]+)/beta([\d.]+)_U(\d+)"
        for file in Path(path).rglob('*.pickle'):
            if pattern in str(file):
                # print(file)
                pickle_path = file
                
    if pickle_path is None:
        raise FileNotFoundError(f"No pickle found for nflux={nflux}, n={n}, U={U}, beta={beta} in {path}")
        
    # Load sig
    with open(pickle_path, 'rb') as file:
        sig = pickle.load(file)
    return sig

############################ Other random helper funcs ################################

def find_data_folder(dir, nflux, n, U, beta):
    """Find data dir with given params in dir (e.g. 8x8_tp0)"""
    search_pattern = os.path.join(
        dir,
        f"nflux{nflux}",
        f"n{n}",
        f"beta{beta:g}_U{U}_mu*"
    )
    matches = glob.glob(search_pattern)
    if matches:
        return matches[0] + '/'  # Return the first match
    return None

def find_nearest(array, value, get_idx = False):
    diff_arr = array - value
    if array.ndim == 1:
        diff_mag_arr = np.abs(diff_arr)
    else:
        diff_mag_arr = np.linalg.norm(diff_arr, axis=-1) # low key questionable lmao
    idx = (diff_mag_arr).argmin()
    diff = value - array[idx]
    if get_idx:
        return idx
    else:
        return array[idx]