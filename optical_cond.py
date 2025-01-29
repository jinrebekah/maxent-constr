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

        self.jj, self.sign, self.n_sample, self.n_bin = self._load_data(path) # note: sign and jj are already divided by n_sample
        self.jjq0, self.chi_xx, self.chi_xy = self._prep_jjq0()

        # Set solver settings (stupid)
        settings_xx_default = {'mdl': 'flat', 'krnl': 'symm', 'opt_method': 'Bryan'}
        self.settings_xx = {**settings_xx_default, **settings_xx}
        self.input_xx = self._get_settings_vals(self.settings_xx)

        settings_xy_default = {'mdl': 'flat', 'opt_method': 'Bryan'}
        self.settings_xy = {**settings_xy_default, **settings_xy}
        self.input_xy = self._get_settings_vals(self.settings_xy)

        # Initialize sigma results storage
        self.results = {
            'xx': {
                'bs': {
                    'A_xx': None,
                    're_sig_xx': None,
                    're_sig_xx_mean': None,
                    're_sig_xx_std': None    
                },
                'all_bins': {
                    'A_xx': None,
                    're_sig_xx': None
                }
            },
            'xy': {
                'bs': {
                    'A_sum': None,
                    'sig_sum': None,
                    'im_sig_xy': None,
                    're_sig_xy': None,
                    're_sig_xy_mean': None,
                    're_sig_xy_std': None
                },
                'all_bins': {
                    'A_sum': None,
                    'sig_sum': None,
                    'im_sig_xy': None,
                    're_sig_xy': None
                }
            }
        }
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
            krnl = maxent.kernel_b(self.beta, self.taus[0 : self.L // 2 + 1], self.ws, sym=True)
        else:
            krnl = maxent.kernel_b(self.beta, self.taus[:-1], self.ws, sym=False)
        opt_method = settings['opt_method']
        return {'mdl': mdl, 'krnl': krnl, 'opt_method': opt_method}

    def calc_sigma_xx(self):
        if self.bs:
            A_xx_bs = np.zeros((self.bs, self.N))
            re_sigmas_xx_bs = np.zeros((self.bs, self.N))
            for i in range(self.bs):
                resample = np.random.randint(0, self.n_bin, self.n_bin)
                re_sigmas_xx_bs[i], A_xx_bs[i] = self._calc_sigma_xx_bins(resample)
            self.results['xx']['bs']['re_sig_xx'], self.results['xx']['bs']['A_xx'] = re_sigmas_xx_bs, A_xx_bs
            self.results['xx']['bs']['re_sig_xx_mean'], self.results['xx']['bs']['re_sig_xx_std'] = np.mean(re_sigmas_xx_bs, axis=0), np.std(re_sigmas_xx_bs, axis=0)
        else:
            all_bins = np.arange(self.n_bin)
            re_sigmas_xx, A_xx = self._calc_sigma_xx_bins(all_bins)
            self.results['xx']['all_bins']['re_sig_xx'], self.results['xx']['all_bins']['A_xx'] = re_sigmas_xx, A_xx
    
    def _calc_sigma_xx_bins(self, resample):
        """Calculates sigma_xx for bin indices specified by resample."""
        f = self.chi_xx[resample].mean(0)
        chiq0w0 = CubicSpline(self.taus, np.append(f, f[0])).integrate(0, self.beta)
        
        g = 2 * self.chi_xx[resample, : self.L // 2 + 1] / chiq0w0
        A_xx = maxent.maxent(g, self.input_xx['krnl'], self.input_xx['mdl'], opt_method=self.input_xx['opt_method'], inspect=False)
        re_sigmas_xx = np.real(A_xx / self.dws * (chiq0w0 / self.sign.mean()) * np.pi) #### final factor of 2 for range of w centered at (not starting at) 0
        return re_sigmas_xx, A_xx

    def calc_sigma_xy(self):
        self.xs = np.linspace(-np.max(self.ws), np.max(self.ws), 1500) # ws used in Kramer's Kronig transform, change to make specifiable later
        bs = self.bs
        if bs:
            re_sigmas_xy_bs = np.zeros((bs, len(self.xs)))
            im_sigmas_xy_bs, sigmas_sum_bs, A_sum_bs, re_sigmas_xx_bs, A_xx_bs = np.zeros((bs, self.N)), np.zeros((bs, self.N)), np.zeros((bs, self.N)), np.zeros((bs, self.N)), np.zeros((bs, self.N))
            for i in tqdm(range(bs)):
                resample = np.random.randint(0, self.n_bin, self.n_bin)
                re_sigmas_xy_bs[i], im_sigmas_xy_bs[i], sigmas_sum_bs[i], A_sum_bs[i], re_sigmas_xx_bs[i], A_xx_bs[i] = self._calc_sigma_xy_bins(resample)
            self.results['xx']['bs']['re_sig_xx'], self.results['xx']['bs']['A_xx'] = re_sigmas_xx_bs, A_xx_bs
            self.results['xy']['bs']['re_sig_xy'], self.results['xy']['bs']['im_sig_xy'], self.results['xy']['bs']['sig_sum'], self.results['xy']['bs']['A_sum'] = re_sigmas_xy_bs, im_sigmas_xy_bs, sigmas_sum_bs, A_sum_bs
            self.results['xx']['bs']['re_sig_xx_mean'], self.results['xx']['bs']['re_sig_xx_std'] = np.mean(re_sigmas_xx_bs, axis=0), np.std(re_sigmas_xx_bs, axis=0)
            self.results['xy']['bs']['re_sig_xy_mean'], self.results['xy']['bs']['re_sig_xy_std'] = np.mean(re_sigmas_xy_bs, axis=0), np.std(re_sigmas_xy_bs, axis=0)
        else:
            all_bins = np.arange(self.n_bin)
            re_sigmas_xy, im_sigmas_xy, sigmas_sum, A_sum, re_sigmas_xx, A_xx = self._calc_sigma_xy_bins(all_bins)
            self.results['xx']['all_bins']['re_sig_xx'], self.results['xx']['all_bins']['A_xx'] = re_sigmas_xx, A_xx
            self.results['xy']['all_bins']['re_sig_xy'], self.results['xy']['all_bins']['im_sig_xy'], self.results['xy']['all_bins']['sig_sum'], self.results['xy']['all_bins']['A_sum'] = re_sigmas_xy, im_sigmas_xy, sigmas_sum, A_sum
    
    def _calc_sigma_xy_bins(self, resample):
        """Calculates sigma_xy for bin indices specified by resample."""
        # Get sigma_xx
        re_sigmas_xx, A_xx = self._calc_sigma_xx_bins(resample)
        # Maxent sum
        f = np.append(self.chi_xx[resample].mean(0), self.chi_xx[resample].mean(0)[0]) - np.real(1j*np.append(self.chi_xy[resample].mean(0), -self.chi_xy[resample].mean(0)[0]))
        chiq0w0 = CubicSpline(self.taus, f).integrate(0, self.beta)
        g = (self.chi_xx[resample] - np.real(1j*self.chi_xy[resample])) / chiq0w0
        if self.input_xy['opt_method'] == 'Bryan':
            A_sum = maxent.maxent(g, self.input_xy['krnl'], self.input_xy['mdl'], opt_method="Bryan", inspect=False)
        elif self.input_xy['opt_method'] == 'cvxpy':
            # Define symmetry constraint matrices
            b = 2*A_xx[self.N//2:]
            B = np.hstack((np.flip(np.identity(self.N//2), axis=0), np.identity(self.N//2)))
            A_sum = maxent.maxent(g, self.input_xy['krnl'], self.input_xy['mdl'], opt_method='cvxpy', constr_matrix=B, constr_vec=b, inspect=False)
        sigmas_sum = np.real(A_sum / self.dws * (chiq0w0 / self.sign[resample].mean())) * np.pi
        im_sigmas_xy = sigmas_sum-re_sigmas_xx

        # Kramer's Kronig for re_sigma_xy
        ys = CubicSpline(self.ws, im_sigmas_xy)(self.xs)
        re_sigmas_xy = -np.imag(scipy.signal.hilbert(ys))
        return re_sigmas_xy, im_sigmas_xy, sigmas_sum, A_sum, re_sigmas_xx, A_xx

    
    def plot_sigma(self, sigma_type, plot_bs=False, bs_mode='errorbar'):
        # Fix, very rudimentary
        #  Maybe later make an option to do things side by side
        sigma_name_dict = {
            "re_sig_xx": r'Re[$\sigma_{xx}(\omega)$]', 
            "im_sig_xx": r'Im[$\sigma_{xx}(\omega)$]',
            "re_sig_xy": r'Re[$\sigma_{xy}(\omega)$]',
            "im_sig_xy": r'Im[$\sigma_{xy}(\omega)$]',
            "sig_sum": r'Re[$\sigma_{xx}(\omega)$] + Im[$\sigma_{xy}(\omega)$]'
        }
        sigma_cat = 'xx' if 'xx' in sigma_type else 'xy'
        if sigma_type == 're_sig_xy':
            ws = self.xs
        else:
            ws = self.ws

        fig, ax = plt.subplots()
        if plot_bs:
            if bs_mode=='errorbar':
                # Plot bootstrap mean with std error bars
                sig = self.results[sigma_cat]['bs'][sigma_type+'_mean']
                sig_err = self.results[sigma_cat]['bs'][sigma_type+'_std']
                ax.errorbar(ws, sig, yerr=sig_err, fmt='s-', lw=0.7, ms=0, capsize=0, ecolor='orange', elinewidth=0.5)
            else:
                # Plot all bootstraps on top of each other
                sig_bs = self.results[sigma_cat]['bs'][sigma_type]
                for i in range(self.bs):
                    ax.plot(ws, sig_bs[i], lw=1, color='#0C5DA5', alpha=0.9)
        else:
            # Plot all bins result
            sig = self.results['xx' if 'xx' in sigma_type else 'xy']['all_bins'][sigma_type]
            ax.plot(ws, sig)
        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel(sigma_name_dict[sigma_type])
        ax.set_title(rf'U = {self.U}, $\beta$ = {self.beta}')
        plt.show()
    
    def print_summary(self):
        # Print summary of settings used in opt
        pass

    def check_chi_xy(self):
        """Multiply im_sig_xy result by K and check residuals against chi_xy."""
        pass

    # def plot_chi_xy(self):
    #     fig, ax = plt.subplots(figsize=(6, 2), ncols=3, layout='constrained')
    #     titles = [r"$ \langle j_x j_y \rangle $", r"$ \langle j_y j_x \rangle $", r"$\chi_{xy}$ (tot)"]
    #     ax[0].plot(np.mean(np.imag(jxjyq0), axis=0))
    #     ax[1].plot(np.mean(np.imag(jyjxq0), axis=0))
    #     ax[2].plot(np.mean(np.imag(chi_xy), axis=0))
    #     for i in range(3): 
    #         ax[i].set_title(titles[i])
    #         ax[i].set_xlabel(r'$\tau$')
    #     plt.suptitle(rf"U = {U}, $\beta$ = {beta}")