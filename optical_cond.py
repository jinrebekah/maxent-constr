import numpy as np
import matplotlib.pyplot as plt
import scipy

import my_my_maxent as maxent
import importlib
importlib.reload(maxent)
import sys
# sys.path.append('/oak/stanford/orgs/simes/rebjin/dqmc-dev/util')
sys.path.append('/Users/rebekahjin/Documents/Devereaux Group/dqmc-dev/util')
import util
from tqdm import tqdm
import math
import pandas as pd
import seaborn as sns

from scipy.interpolate import CubicSpline
default_figsize = plt.rcParams['figure.figsize']

class sigma:
    def __init__(self, path, sigma_type, ws, dws, bs=0, settings_xx = {}, settings_xy={}):
        # Store simulation parameters
        self.U, self.Ny, self.Nx, self.beta, self.L, self.tp = util.load_firstfile(
            path, "metadata/U", "metadata/Nx", "metadata/Ny", "metadata/beta", "params/L", "metadata/t'"
        )
        self.T = 1/self.beta
        self.taus = np.linspace(0, self.beta, self.L + 1)

        self.ws = ws
        self.dws = dws
        self.N = len(ws)
        self.bs = bs
        self.sigma_type = sigma_type

        self.jj, self.sign, self.n_sample, self.n_bin = self._load_data(path) # note: sign and jj are already divided by n_sample
        self.jjq0, self.chi_xx, self.chi_xy = self._prep_jjq0()

        # Set solver settings (stupid)
        settings_xx_default = {
            'mdl': 'flat', 
            'krnl': 'symm', 
            'opt_method': 'Bryan',
            'inspect_al': False
        }
        self.settings_xx = {**settings_xx_default, **settings_xx}
        self.input_xx = self._get_settings_vals(self.settings_xx)

        settings_xy_default = {
            'mdl': 'flat',
            'opt_method': 'Bryan',
            'inspect_al': False
        }
        self.settings_xy = {**settings_xy_default, **settings_xy}
        self.input_xy = self._get_settings_vals(self.settings_xy)

        # Initialize sigma results storage df
        results = pd.DataFrame(columns=['re_sig_xx','A_xx', 'norm', 'al'])
        if sigma_type == 'xy':
            results[['sum_sig','A_sum', 'im_sig_xy', 're_sig_xy']] = [None] * 4

        # Solve for sigma
        if sigma_type == 'xx':
            self.calc_sigma_xx()
        if sigma_type == 'xy':
            self.calc_sigma_xy()

    def _load_data(self, path):
        """Loads j-j data."""
        # Load in all measurements
        n_samples, sign, jj = util.load(
            path, "meas_uneqlt/n_sample", "meas_uneqlt/sign", "meas_uneqlt/jj"
        )
        # Only keep measurements from bins where n_sample is max value (i.e., bin is complete)
        n_sample = n_samples.max()
        mask = n_samples == n_sample
        sign, jj = sign[mask], jj[mask]
        n_bin = mask.sum()
        # Reshape jj into a 2x2 matrix (or 4x4, for nonzero t') discretized in tau + containing info from bond to bond (indexed by lattice site)
        jj.shape = -1, self.L, 2, 2, self.Ny, self.Nx
        # I guess also note that jj does not include tau=beta, remember len(taus) != L
        return jj/n_sample, sign/n_sample, n_sample, n_bin

    def _prep_jjq0(self):
        """Gets jjq0 and averaged xx and xy correlators."""
        # Will probably need to change, this only works for tp=0
        jjq0 = self.jj.sum((-1, -2))   # sum over all the bonds (since q=0)
        # Get average longitudinal jj
        jxjxq0 = -jjq0[..., 0, 0]
        jyjyq0 = -jjq0[..., 1, 1]
        chi_xx = 0.5 * (jxjxq0 + jyjyq0) # average over xx and yy to get avg longitudinal j-j
        chi_xx = 0.5 * (chi_xx + chi_xx[:, -np.arange(self.L) % self.L]) # symmetrize bin by bin
        chi_xx = np.real(chi_xx) # added for nflux != 0 data, should be purely real
        # Get average transverse jj
        jxjyq0 = jjq0[..., 1, 0]
        jyjxq0 = -jjq0[..., 0, 1]
        chi_xy = 0.5*(jxjyq0+jyjxq0)
        chi_xy = np.concatenate((np.expand_dims(chi_xy[:, 0], axis=1), 0.5*(chi_xy[:, 1:] - chi_xy[:, :0:-1])), axis=1) # stupid but antisymmetrize bin by bin
        chi_xy = 1j*np.imag(chi_xy) # added for nflux != 0 data, should be purely imaginary
        return jjq0, chi_xx, chi_xy
    
    def _get_settings_vals(self, settings):
        """Returns input dict of parameters directly passed to MaxEnt."""
        mdl = maxent.model_flat(self.dws) if settings['mdl'] == 'flat' else settings['mdl']
        if 'krnl' in settings and settings['krnl'] == 'symm':
            krnl = maxent.kernel_b(self.beta, self.taus[0 : self.L // 2 + 1], self.ws[self.N//2:], sym=True)
            mdl = mdl[self.N//2:]
        else:
            krnl = maxent.kernel_b(self.beta, self.taus[:-1], self.ws, sym=False)
        opt_method = settings['opt_method']
        # inspect_al = settings['inspect_al'] if self.bs==0 else False # overrides input, can only be true for bs = 0
        smooth_al = True if settings['opt_method'] == 'cvxpy' else False
        als = np.logspace(8, 1, 1+20*(8-1)) if 'krnl' in settings else np.logspace(8, 2, 1+20*(8-2))
        return {'m': mdl, 'K': krnl, 'opt_method': opt_method, 'smooth_al': smooth_al, 'als': als}

    def calc_sigma_xx(self):
        if self.bs:
            bs_list = []
            for i in tqdm(range(self.bs), desc='Sigma_xx bootstraps'):
                resample = np.random.randint(0, self.n_bin, self.n_bin)
                re_sigmas_xx, debug_vals = self._calc_sigma_xx_bins(resample)
                bs_dict = {'re_sig_xx': re_sigmas_xx, 'resample': resample, **debug_vals}
                bs_list.append(bs_dict)
        else:
            all_bins = np.arange(self.n_bin)
            re_sigmas_xx, debug_vals = self._calc_sigma_xx_bins(all_bins, inspect_al = self.settings_xx['inspect_al'])
            bs_list = [{'re_sig_xx': re_sigmas_xx, 'resample': all_bins, **debug_vals}]
        # Create results dataframe from list of bs dicts
        self.results = pd.DataFrame(bs_list)
    
    def _calc_sigma_xx_bins(self, resample, inspect_al=False):
        """Calculates sigma_xx for bin indices specified by resample."""
        f = self.chi_xx[resample].mean(0)
        chiq0w0 = CubicSpline(self.taus, np.append(f, f[0])).integrate(0, self.beta)
        if self.settings_xx['krnl'] == 'symm':
            # Symmetric krnl, with half tau and w range. Only unconstrained option
            g = self.chi_xx[resample, : self.L // 2 + 1] / chiq0w0 # when we truncate taus, it includes the midpoint
            A_xx, al_xx, As_xx, chi2s_xx = maxent.maxent(g, **self.input_xx) # No factor of 2 here
            # Fill in the negative w half of A_xx
            A_xx = np.concatenate((A_xx[::-1], A_xx))
        else:
            # Full krnl
            g = self.chi_xx[resample] / chiq0w0
            if self.input_xx['opt_method'] == 'Bryan':
                # Unconstrained
                A_xx, al_xx, As_xx, chi2s_xx = maxent.maxent(g, **self.input_xx, inspect_al=inspect_al)
            else:
                # Define symmetry constraint matrices for A_xx
                b = np.zeros(self.N//2)
                B = np.hstack((np.flip(np.identity(self.N//2), axis=0), -1*np.identity(self.N//2)))
                self.input_xx['constr_matrix'] = B
                self.input_xx['constr_vec'] = b
                A_xx, al_xx, As_xx, chi2s_xx = maxent.maxent(g, **self.input_xx, inspect_al=inspect_al)
        re_sigmas_xx = np.real(A_xx / self.dws * (chiq0w0 / self.sign.mean()) * np.pi)
        debug_vals = {'A_xx': A_xx, 'norm_xx': chiq0w0, 'al_xx': al_xx, 'As_xx': As_xx, 'chi2s_xx': chi2s_xx}
        return re_sigmas_xx, debug_vals

    def calc_sigma_xy(self):
        self.xs = np.linspace(-np.max(self.ws), np.max(self.ws), 1500) # ws used in Kramer's Kronig transform, change to make specifiable later
        if self.bs:
            bs_list = []
            for i in tqdm(range(self.bs), desc='Sigma_xy bootstraps'):
                resample = np.random.randint(0, self.n_bin, self.n_bin)
                re_sigmas_xy, im_sigmas_xy, sigmas_sum, re_sigmas_xx, debug_vals = self._calc_sigma_xy_bins(resample)
                bs_dict = {'re_sig_xx': re_sigmas_xx, 'im_sig_xy': im_sigmas_xy, 'sig_sum': sigmas_sum, 're_sig_xy': re_sigmas_xy, 'resample': resample, **debug_vals}
                bs_list.append(bs_dict)
        else:
            all_bins = np.arange(self.n_bin)
            re_sigmas_xy, im_sigmas_xy, sigmas_sum, re_sigmas_xx, debug_vals = self._calc_sigma_xy_bins(all_bins, inspect_al = self.settings_xy['inspect_al'])
            bs_list = [{'re_sig_xx': re_sigmas_xx, 'im_sig_xy': im_sigmas_xy, 'sig_sum': sigmas_sum, 're_sig_xy': re_sigmas_xy, 'resample': all_bins, **debug_vals}]
        # Create results dataframe from list of bs dicts
        self.results = pd.DataFrame(bs_list)

    def _calc_sigma_xy_bins(self, resample, inspect_al=False):
        """Calculates sigma_xy for bin indices specified by resample."""
        # Get sigma_xx
        re_sigmas_xx, debug_vals_xx = self._calc_sigma_xx_bins(resample)
        A_xx = debug_vals_xx['A_xx']
        # Maxent sum
        f = np.append(self.chi_xx[resample].mean(0), self.chi_xx[resample].mean(0)[0]) - np.real(1j*np.append(self.chi_xy[resample].mean(0), -self.chi_xy[resample].mean(0)[0]))
        chiq0w0 = CubicSpline(self.taus, f).integrate(0, self.beta)
        g = (self.chi_xx[resample] - np.real(1j*self.chi_xy[resample])) / chiq0w0
        if self.input_xy['opt_method'] == 'Bryan':
            # Unconstrained
            A_sum, al_sum, As_sum, chi2s_sum = maxent.maxent(g, **self.input_xy, inspect_al = inspect_al)
        elif self.input_xy['opt_method'] == 'cvxpy':
            # Define symmetry constraint matrices
            b = 2*A_xx[self.N//2:]
            B = np.hstack((np.flip(np.identity(self.N//2), axis=0), np.identity(self.N//2)))
            self.input_xy['constr_matrix'] = B
            self.input_xy['constr_vec'] = b
            A_sum, al_sum, As_sum, chi2s_sum = maxent.maxent(g, **self.input_xy, inspect_al = inspect_al)
        sigmas_sum = np.real(A_sum / self.dws * (chiq0w0 / self.sign[resample].mean())) * np.pi
        im_sigmas_xy = sigmas_sum-re_sigmas_xx
        # Kramer's Kronig for re_sigma_xy
        ys = CubicSpline(self.ws, im_sigmas_xy)(self.xs)
        re_sigmas_xy = -np.imag(scipy.signal.hilbert(ys))

        debug_vals = {'norm_sum': chiq0w0, 'A_sum': A_sum, 'A_xy': A_sum-A_xx, 'al_sum': al_sum, 'As_sum': As_sum, 'chi2s_sum': chi2s_sum, **debug_vals_xx}
        return re_sigmas_xy, im_sigmas_xy, sigmas_sum, re_sigmas_xx, debug_vals

    def print_summary(self):
        # Print summary of settings used in opt
        pass

    def get_chi_xx(self, include_beta=True):
        '''Reproduces G_xx(tau)'''
        A_xx = (self.results['A_xx']*self.results['norm_xx']).mean()

        chi_xx = np.mean(self.chi_xx, axis=0)
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
    
    def get_chi_xy(self, include_beta=True):
        '''Reproduces G_xy(tau)'''
        # Get from df
        # If bs, just average A_xy*norm for all bs
        A_xy = (self.results['A_xy']*self.results['norm_sum']).mean()

        KA = self.input_xy['K']@A_xy
        chi_xy = np.mean(self.chi_xy, axis=0)
        if include_beta:    
            chi_xy = np.append(chi_xy, -chi_xy[0])
            KA = np.append(KA, -KA[0])
        return KA, chi_xy


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
        sig.plot_sigma(sig, ax, sig_names[0], bs_idx, bs_mode=bs_mode)
    else:
        for i in range(num_plots): plot_sigma(sig, ax[i], sig_names[i], bs_idx=bs_idx, bs_mode=bs_mode)
    
    fig.suptitle(rf'U = {sig.U}, $\beta$ = {sig.beta}, bs = {sig.bs}')
    # plt.tight_layout()
    plt.show()

def plot_sigma(sig, ax, sigma_name, bs_idx=None, bs_mode='errorbar'):
    sigma_name_dict = {
        "re_sig_xx": r'Re[$\sigma_{xx}(\omega)$]', 
        # "im_sig_xx": r'Im[$\sigma_{xx}(\omega)$]',
        "re_sig_xy": r'Re[$\sigma_{xy}(\omega)$]',
        "im_sig_xy": r'Im[$\sigma_{xy}(\omega)$]',
        "sig_sum": r'Re[$\sigma_{xx}(\omega)$] + Im[$\sigma_{xy}(\omega)$]'
    }

    if sigma_name == 're_sig_xy':
        ws = sig.xs
    else:
        ws = sig.ws

    if sig.bs:
        if bs_idx is None:
            # Plot all bootstraps
            bs_idx = np.arange(sig.bs) # bs_idx needs to be a list
        sig_bs = np.array(sig.results[sigma_name].tolist())[bs_idx] # Only keep bs we want to plot
        if len(sig_bs) == 1:
            bs_mode = 'all'   # no such thing as std for 1 bs, use 'all' mode
        if bs_mode=='errorbar':
            # Plot bootstrap mean with std error bars
            ax.errorbar(ws, np.mean(sig_bs, axis=0), yerr=np.std(sig_bs, axis=0), fmt='s-', lw=0.7, ms=0, capsize=0, ecolor='orange', elinewidth=0.5)
        else:
            # Plot all bootstraps on top of each other
            for i in range(len(sig_bs)):
                ax.plot(ws, sig_bs[i], lw=1, color='#0C5DA5', alpha=0.9)
    else:
        # Plot all bins result
        ax.plot(ws, sig.results[sigma_name][0])

    # Annotate with opt info in top left corner I guess
    settings = sig.settings_xx if 'xx' in sigma_name else sig.settings_xy
    method = settings['opt_method']
    K = sig.settings_xx['krnl']
    # al_method = 'smooth' if settings['smooth_al'] else 'default'
    ax.annotate('O: ' + method + '\n' + r'$K_{xx}$: ' + K, (0.04, 0.84), xycoords='axes fraction', fontsize=8, color='gray')
    # ax.annotate(f'O: {opt_method_dict[method]} \n$K_{xx}$: {K}', (0.03, 0.89), xycoords='axes fraction')
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(sigma_name_dict[sigma_name])

    # ax.set_title(rf'U = {sig.U}, $\beta$ = {sig.beta}')

def inspect_al(sig, sigma_type, bs, redo_select_al = False, als_plot=[]):
    # Jk actually just redo the bootstrap essentially lmfao just to see the alpha selection plot
    # Also include color plot of spectra vs. alpha
    resample = sig.results['resample'][bs]
    
    if sig.settings_xx['krnl'] == 'symm':
        As_xx = np.concatenate((sig.results['As_xx'][bs][:, ::-1], sig.results['As_xx'][bs]), axis=1)
    else:
        As_xx = sig.results['As_xx'][bs]

    if sigma_type == 'xx':
        # See color plot of sig_xx spectra vs. alphas
        if redo_select_al: sig._calc_sigma_xx_bins(resample, inspect_al = True)
        As = sig.results['As_xx'][bs]
        sigmas_xx_al = np.real(As_xx/sig.dws * (sig.results['norm_xx'][bs]/sig.sign[resample].mean())*np.pi)
        sigmas_al = sigmas_xx_al
        optimal_al = sig.results['al_xx'][bs]
        chi2s = sig.results['chi2s_xx'][bs]
        sig_label = r'Re[$\sigma_{xx}(\omega)$]'
        als = sig.input_xx['als']
    else:
        # See color plot of im_sig_xy vs. alphas
        if redo_select_al: sig._calc_sigma_xy_bins(resample, inspect_al = True)
        # sigmas_xx_al = As_xx/sig.dws * (sig.results['norm_xx'][bs]/sig.sign[resample].mean())*np.pi # not necessary
        As = sig.results['As_sum'][bs]
        sigmas_xx = sig.results['re_sig_xx'][bs]

        sigmas_sum_al = np.real((sig.results['As_sum'][bs])/sig.dws * (sig.results['norm_sum'][bs]/sig.sign[resample].mean())*np.pi)
        sigmas_al = sigmas_sum_al - sigmas_xx
        optimal_al = sig.results['al_sum'][bs]
        chi2s = sig.results['chi2s_sum'][bs]
        sig_label = r'Im[$\sigma_{xy}(\omega)$]'
        als = sig.input_xy['als']
    als_plot.append(optimal_al) # always plot optimal al

    # Plot density plot of sigma vs. al, with neighboring plot of spectra at alpha slices in als_plot
    fig, ax = plt.subplots(figsize = (default_figsize[0]*3, default_figsize[1]), ncols=3, layout='constrained')

    # Chi2 plot
    ax[0].scatter(als, chi2s)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$\alpha$')
    ax[0].set_ylabel(r'$\chi^2$')

    # Color plot
    lim = max(np.nanmin(sigmas_al), np.nanmax(sigmas_al))/2
    print(lim)
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    X, Y = np.meshgrid(als, sig.ws)
    pcol = ax[1].pcolormesh(X, Y, np.transpose(sigmas_al), cmap='plasma', rasterized=True, norm=norm)
    ax[1].invert_yaxis()
    ax[1].set_xscale('log')
    fig.colorbar(pcol, ax=ax[1])
    ax[1].set_xlabel(r'$\alpha$')
    ax[1].set_ylabel(r'$\omega$')
    ax[1].set_ylim(-20, 20)

    # Spectrum plot
    if len(als_plot) == 1:
        colors = ['r']
    else:
        colors = sns.color_palette('tab10', len(als_plot)-1)
        colors.append('r')
    for i, al_plot in enumerate(als_plot):
        color = colors[i]
        al_idx = find_nearest(als, al_plot, get_idx=True)
        ax[2].plot(sig.ws, sigmas_al[al_idx], color=color, label=rf'$\alpha$ = {al_plot: .2e}')
        for j in range(2): ax[j].axvline(al_plot, color=color) # Plot lines on colorplot and chi2 plots at als_plot
    ax[2].set_xlabel(r'$\omega$')
    ax[2].set_ylabel(sig_label)
    ax[2].set_xlim(-20, 20)
    ax[2].legend()

    plt.show()

def compare_chi_tau(sigs, mode='xx'):
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
    resids = [KA-chi for KA in KAs]
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = color_cycle[1:3]

    plot_size = plt.rcParams['figure.figsize']
    fig, ax = plt.subplots(ncols=2, figsize=(plot_size[0]*2, plot_size[1]*1.2), layout='constrained')
    ax[0].plot(taus, chi, label=chi_label)
    for i in range(len(sigs)): ax[0].plot(taus, KAs[i], label=labels[i])
    ax[0].set_title('Data')
    ax[0].legend()

    for i in range(len(sigs)): ax[1].scatter(taus, resids[i], color=colors[i], s=7)
    ax[1].axhline(0, color='gray', ls='--', alpha=0.5)
    # ax[1].set_ylabel('Residuals')
    ax[1].set_title('Residuals')

    for i in range(2): ax[i].set_xlabel(r'$\tau$')
    fig.suptitle(rf'U = {U}, $\beta$ = {beta}, bs = {bs}')
    # plt.tight_layout()
    plt.show()

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