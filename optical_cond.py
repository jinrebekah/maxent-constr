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
        settings_xx_default = {'mdl': 'flat', 'krnl': 'symm', 'opt_method': 'Bryan'}
        self.settings_xx = {**settings_xx_default, **settings_xx}
        self.input_xx = self._get_settings_vals(self.settings_xx)

        settings_xy_default = {'mdl': 'flat', 'opt_method': 'Bryan'}
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
        # Possibly better to do real() and imag() after Maxent??
        # Get average transverse jj
        jxjyq0 = jjq0[..., 1, 0]
        jyjxq0 = -jjq0[..., 0, 1]
        chi_xy = 0.5*(jxjyq0+jyjxq0)
        chi_xy = np.concatenate((np.expand_dims(chi_xy[:, 0], axis=1), 0.5*(chi_xy[:, 1:] - chi_xy[:, :0:-1])), axis=1) # stupid but antisymmetrize bin by bin
        chi_xy = 1j*np.imag(chi_xy) # added for nflux != 0 data, should be purely imaginary
        return jjq0, chi_xx, chi_xy
    
    def _get_settings_vals(self, settings):
        """Kinda dumb but this generates a dict with values corresponding to settings dict."""
        mdl = maxent.model_flat(self.dws) if settings['mdl'] == 'flat' else settings['mdl']
        if 'krnl' in settings and settings['krnl'] == 'symm':
            krnl = maxent.kernel_b(self.beta, self.taus[0 : self.L // 2 + 1], self.ws[self.N//2:], sym=True)
            mdl = mdl[self.N//2:]
        else:
            krnl = maxent.kernel_b(self.beta, self.taus[:-1], self.ws, sym=False)
        opt_method = settings['opt_method']
        return {'mdl': mdl, 'krnl': krnl, 'opt_method': opt_method}

    def calc_sigma_xx(self):
        if self.bs:
            bs_list = []
            for i in tqdm(range(self.bs)):
                resample = np.random.randint(0, self.n_bin, self.n_bin)
                re_sigmas_xx, debug_vals = self._calc_sigma_xx_bins(resample)
                bs_dict = {'re_sig_xx': re_sigmas_xx, **debug_vals}
                bs_list.append(bs_dict)
        else:
            all_bins = np.arange(self.n_bin)
            re_sigmas_xx, debug_vals = self._calc_sigma_xx_bins(all_bins)
            bs_list = [{'re_sig_xx': re_sigmas_xx, **debug_vals}]

        # Create results dataframe from list of bs dicts
        self.results = pd.DataFrame(bs_list)
    
    def _calc_sigma_xx_bins(self, resample):
        """Calculates sigma_xx for bin indices specified by resample."""
        f = self.chi_xx[resample].mean(0)
        chiq0w0 = CubicSpline(self.taus, np.append(f, f[0])).integrate(0, self.beta)
        if self.settings_xx['krnl'] == 'symm':
            # Adjusted factors of 2 bc this is stupid
            g = self.chi_xx[resample, : self.L // 2 + 1] / chiq0w0 # when we truncate taus, it includes the midpoint
            ws_maxent = self.ws[self.N//2:]   # only positive range of ws (N should be an even number)
            A_xx, al_xx = maxent.maxent(g, self.input_xx['krnl'], self.input_xx['mdl'], opt_method=self.input_xx['opt_method']) # No factor of 2 here
            # Fill in the negative w half of A_xx
            A_xx = np.concatenate((A_xx[::-1], A_xx))
        else:
            g = self.chi_xx[resample] / chiq0w0
            if self.input_xx['opt_method'] == 'Bryan':
                A_xx, al_xx = maxent.maxent(g, self.input_xx['krnl'], self.input_xx['mdl'], opt_method=self.input_xx['opt_method'])
            else:
                # Define symmetry constraint matrices for A_xx
                b = np.zeros(self.N//2)
                B = np.hstack((np.flip(np.identity(self.N//2), axis=0), -1*np.identity(self.N//2)))
                A_xx, al_xx = maxent.maxent(g, self.input_xx['krnl'], self.input_xx['mdl'], opt_method=self.input_xx['opt_method'], constr_matrix=B, constr_vec=b)
        re_sigmas_xx = np.real(A_xx / self.dws * (chiq0w0 / self.sign.mean()) * np.pi)

        debug_vals = {'A_xx': A_xx, 'norm_xx': chiq0w0, 'al_xx': al_xx}
        return re_sigmas_xx, debug_vals

    def calc_sigma_xy(self):
        self.xs = np.linspace(-np.max(self.ws), np.max(self.ws), 1500) # ws used in Kramer's Kronig transform, change to make specifiable later
        if self.bs:
            bs_list = []
            for i in tqdm(range(self.bs)):
                resample = np.random.randint(0, self.n_bin, self.n_bin)
                re_sigmas_xy, im_sigmas_xy, sigmas_sum, re_sigmas_xx, debug_vals = self._calc_sigma_xy_bins(resample)
                bs_dict = {'re_sig_xx': re_sigmas_xx, 'im_sig_xy': im_sigmas_xy, 'sig_sum': sigmas_sum, 're_sig_xy': re_sigmas_xy, **debug_vals}
                bs_list.append(bs_dict)
        else:
            all_bins = np.arange(self.n_bin)
            re_sigmas_xy, im_sigmas_xy, sigmas_sum, re_sigmas_xx, debug_vals = self._calc_sigma_xy_bins(all_bins)
            bs_list = [{'re_sig_xx': re_sigmas_xx, 'im_sig_xy': im_sigmas_xy, 'sig_sum': sigmas_sum, 're_sig_xy': re_sigmas_xy, **debug_vals}]
        # Create results dataframe from list of bs dicts
        self.results = pd.DataFrame(bs_list)

    def _calc_sigma_xy_bins(self, resample):
        """Calculates sigma_xy for bin indices specified by resample."""
        # Get sigma_xx
        re_sigmas_xx, debug_vals_xx = self._calc_sigma_xx_bins(resample)
        A_xx = debug_vals_xx['A_xx']
        # Maxent sum
        f = np.append(self.chi_xx[resample].mean(0), self.chi_xx[resample].mean(0)[0]) - np.real(1j*np.append(self.chi_xy[resample].mean(0), -self.chi_xy[resample].mean(0)[0]))
        chiq0w0 = CubicSpline(self.taus, f).integrate(0, self.beta)
        g = (self.chi_xx[resample] - np.real(1j*self.chi_xy[resample])) / chiq0w0
        if self.input_xy['opt_method'] == 'Bryan':
            A_sum, al_sum = maxent.maxent(g, self.input_xy['krnl'], self.input_xy['mdl'], opt_method="Bryan", inspect_opt=False)
        elif self.input_xy['opt_method'] == 'cvxpy':
            # Define symmetry constraint matrices
            b = 2*A_xx[self.N//2:]
            B = np.hstack((np.flip(np.identity(self.N//2), axis=0), np.identity(self.N//2)))
            A_sum, al_sum = maxent.maxent(g, self.input_xy['krnl'], self.input_xy['mdl'], opt_method='cvxpy', constr_matrix=B, constr_vec=b)
        sigmas_sum = np.real(A_sum / self.dws * (chiq0w0 / self.sign[resample].mean())) * np.pi
        im_sigmas_xy = sigmas_sum-re_sigmas_xx

        # Kramer's Kronig for re_sigma_xy
        ys = CubicSpline(self.ws, im_sigmas_xy)(self.xs)
        re_sigmas_xy = -np.imag(scipy.signal.hilbert(ys))

        debug_vals = {'norm_sum': chiq0w0, 'A_sum': A_sum, 'A_xy': A_sum-A_xx, 'al_sum': al_sum, **debug_vals_xx}
        return re_sigmas_xy, im_sigmas_xy, sigmas_sum, re_sigmas_xx, debug_vals

    def plot_results(self, sig_names='all', bs_mode='errorbar'):
        # Change this to make it plot everything low key, depending on the calculation run
        if sig_names=='all':
            if self.sigma_type == 'xx':
                sig_names = ['re_sig_xx']
            else:
                sig_names = ['re_sig_xx', 'sig_sum', 'im_sig_xy', 're_sig_xy']
        
        num_plots = len(sig_names)
        plot_size = plt.rcParams['figure.figsize']

        fig, ax = plt.subplots(ncols=num_plots, figsize=(plot_size[0]*num_plots, plot_size[1]))
        if num_plots==1:
            self.plot_sigma(ax, sig_names[0], bs_mode=bs_mode)
        else:
            for i in range(num_plots): self.plot_sigma(ax[i], sig_names[i], bs_mode=bs_mode)
        
        fig.suptitle(rf'U = {self.U}, $\beta$ = {self.beta}, bs = {self.bs}')
        plt.tight_layout()
        plt.show()

    def plot_sigma(self, ax, sigma_name, bs_mode='errorbar'):
        sigma_name_dict = {
            "re_sig_xx": r'Re[$\sigma_{xx}(\omega)$]', 
            "im_sig_xx": r'Im[$\sigma_{xx}(\omega)$]',
            "re_sig_xy": r'Re[$\sigma_{xy}(\omega)$]',
            "im_sig_xy": r'Im[$\sigma_{xy}(\omega)$]',
            "sig_sum": r'Re[$\sigma_{xx}(\omega)$] + Im[$\sigma_{xy}(\omega)$]'
        }
        if sigma_name == 're_sig_xy':
            ws = self.xs
        else:
            ws = self.ws

        if self.bs:
            sig_bs = np.array(self.results[sigma_name].tolist())
            if bs_mode=='errorbar':
                # Plot bootstrap mean with std error bars
                sig = np.mean(sig_bs, axis=0)
                sig_err = np.std(sig_bs, axis=0)
                ax.errorbar(ws, sig, yerr=sig_err, fmt='s-', lw=0.7, ms=0, capsize=0, ecolor='orange', elinewidth=0.5)
            else:
                # Plot all bootstraps on top of each other
                for i in range(self.bs):
                    ax.plot(ws, sig_bs[i], lw=1, color='#0C5DA5', alpha=0.9)
        else:
            # Plot all bins result
            sig = self.results[sigma_name][0]
            ax.plot(ws, sig)

        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel(sigma_name_dict[sigma_name])
        # ax.set_title(rf'U = {self.U}, $\beta$ = {self.beta}')
    
    def print_summary(self):
        # Print summary of settings used in opt
        pass

    def get_chi_xx(self, include_beta=True):
        '''Reproduces G_xx(tau)'''
        A_xx = (self.results['A_xx']*self.results['norm_xx']).mean()

        chi_xx = np.mean(self.chi_xx, axis=0)
        if self.settings_xx['krnl']=='symm':
            # A_xx full length, but krnl is not 
            KA = self.input_xx['krnl']@A_xx[self.N//2:]   # only for the first half of taus
            KA = np.concatenate((KA, KA[math.ceil(self.L/2)-1::-1]))[:-1] # without including beta point
        else:
            KA = self.input_xx['krnl']@A_xx
            
        if include_beta:
            KA = np.append(KA, KA[0])
            chi_xx = np.append(chi_xx, chi_xx[0])
        
        return KA, chi_xx
    
    def get_chi_xy(self, include_beta=True):
        '''Reproduces G_xy(tau)'''
        # Get from df
        # If bs, just average A_xy*norm for all bs
        A_xy = (self.results['A_xy']*self.results['norm_sum']).mean()

        KA = self.input_xy['krnl']@A_xy
        chi_xy = np.mean(self.chi_xy, axis=0)
        if include_beta:    
            chi_xy = np.append(chi_xy, -chi_xy[0])
            KA = np.append(KA, -KA[0])
        return KA, chi_xy


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