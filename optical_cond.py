import numpy as np
import matplotlib.pyplot as plt
import scipy

import my_my_maxent as maxent
import importlib
importlib.reload(maxent)
import sys
sys.path.append('/oak/stanford/orgs/simes/rebjin/dqmc-dev/util')
import util

from scipy.interpolate import CubicSpline
default_figsize = plt.rcParams['figure.figsize']

class Sigma:
    # later on, could move the data processing stuff to another function/class but whatever
    def __init__(self, path, sigma_type, ws, dws, mdl, opt_method, bs=0):
        # Store simulation parameters
        # Prob change to a dict called params
        self.Ny, self.Nx, self.beta, self.L, self.tp = util.load_firstfile(
            path, "metadata/Nx", "metadata/Ny", "metadata/beta", "params/L", "metadata/t'"
        )
        self.T = 1/self.beta
        self.taus = np.linspace(0, beta, L + 1)

        self.ws = ws
        self.dws = dws
        self.N = len(ws)
        self.mdl = mdl   # Should probably validate this input at some point so it can be a list
        self.opt_method = opt_method
        self.bs = bs
        
        self.jj, self.sign, self.n_sample, self.n_bin = self._load_data(path) # note: sign and jj are already divided by n_sample
        self.jjq0, self.chi_xx, self.chi_xy = self._prep_jjq0()

        # Calculate sigma
        if sigma_type == 'xx':
            self.calc_sigma_xx()
        elif sigma_type == 'xy':
            # In this case, mdl/opt_method should be list, if str then interpret as same thing for both
            self.calc_sigma_xy()
        else:
            print("Sigma_type invalid.")
        
    def _load_data(self, path):
        # Load in all measurements
        n_samples, sign, jj = util.load(
            path, "meas_uneqlt/n_sample", "meas_uneqlt/sign", "meas_uneqlt/jj"
        )
        # Only keep measurements from bins where n_sample is max value (i.e., bin is complete)
        n_sample = n_samples.max()   
        mask = n_samples == n_sample
        sign, jj = signs[mask], jjs[mask]
        n_bin = mask.sum()
        # Reshape jj into a 2x2 matrix (or 4x4, for nonzero t') discretized in tau + containing info from bond to bond (indexed by lattice site)
        jj.shape = -1, L, 2, 2, Ny, Nx
        # I guess also note that jj does not include tau=beta, remember len(taus) != L
        return jj/n_sample, sign/n_sample, n_sample, n_bin

    def _prep_jjq0(self):
        """Get jjq0 and averaged xx and xy correlators."""
        # Will probably need to change, this only works for tp=0
        jjq0 = self.jj.sum((-1, -2))   # sum over all the bonds (since q=0)
        # Get average longitudinal jj
        jxjxq0 = -jjq0[..., 0, 0]
        jyjyq0 = -jjq0[..., 1, 1]
        chi_xx = 0.5 * (jxjxq0 + jyjyq0) # average over xx and yy to get avg longitudinal j-j
        chi_xx = 0.5 * (chi_xx + chi_xx[:, -np.arange(L) % L]) # symmetrize bin by bin
        chi_xx = np.real(chi_xx) # added for nflux != 0 data, should be purely real
        # Get average transverse jj
        jxjyq0 = jjq0[..., 1, 0]   # ***is this supposed to be minus or plus? idk but clearly one of them should be minus
        jyjxq0 = -jjq0[..., 0, 1]
        chi_xy = 0.5*(jxjyq0+jyjxq0)
        chi_xy = 0.5 * (chi_xy - chi_xy[:, -np.arange(L) % L])
        chi_xy = np.concatenate((np.expand_dims(chi_xy[:, 0], axis=1), 0.5*(chi_xy[:, 1:] - chi_xy[:, :0:-1])), axis=1) # antisymmetrize bin by bin
        chi_xy = 1j*np.imag(chi_xy) # added for nflux != 0 data, should be purely imaginary
        return jjq0, chi_xx, chi_xy
        
    def calc_sigma_xx(self):
        # run_params = {
        #         'sigma_type': 'xx',
        #         'bs': bs,
        #         'opt_method': opt_method,
        #         'symm_ker': symm
        # }
        krnl = maxent.kernel_b(self.beta, self.taus[0 : L // 2 + 1], self.ws, sym=True)
        if self.mdl is None:
            self.mdl = maxent.model_flat(self.dws)
        if self.bs:
            self.A_xx_bs = np.zeros((self.bs, self.N))
            self.re_sigmas_xx_bs = np.zeros((self.bs, self.N))
            for i in range(bs):
                resample = np.random.randint(0, self.nbin, self.nbin)
                self.re_sigmas_xx_bs[i], self.A_xx_bs[i] = self._get_sigma_xx_bins(resample, ws, dws, krnl, mdl, opt_method, symm=symm)
            re_sigmas_xx_mean = np.mean(re_sigmas_xx_bs, axis=0) 
            re_sigmas_xx_std = np.std(re_sigmas_xx_bs, axis=0)
        else:
            all_bins = np.arange(self.n_bin)
            re_sigmas_xx, A_xx = self._get_sigma_xx_bins(all_bins, ws, dws, krnl, mdl, opt_method, symm=symm)
            
        # Create and return a sigma object
        sigmas_xx = sigma(self, ws, dws, run_params, results)
    
    def _get_sigma_xx_bins(self, resample):
        f = self.chi_xx[resample].mean(0)
        chiq0w0 = CubicSpline(self.taus, np.append(f, f[0])).integrate(0, self.beta)
        g = 2 * chi_xx[resample, : self.L // 2 + 1] / chiq0w0
        A_xx = maxent.maxent(g, krnl, self.mdl, opt_method=self.opt_method)
        re_sigmas_xx = A_xx / dws * (chiq0w0 / sign[resample].mean()) * np.pi / 2*2
        return A_xx, re_sigmas_xx

    # def get_sigma_xy(self, ws, dws, mdl, opt_method='Bryan', symm=True, bs=0):
    #     run_params = {
    #             'sigma_type': 'xx',
    #             'bs': bs,
    #             'opt_method': opt_method,
    #             'symm_ker': symm
    #     }
    #     # Maybe we'll handle opt_method and mdl just like before, as a list
    #     krnl = maxent.kernel_b(beta, taus[:-1], ws, sym=False)
    #     if mdl is None:
    #         mdl = maxent.model_flat(dws)
            
    #     if bs:
    #         A_xx_bs = np.zeros((bs, n_w))
    #         re_sigmas_xx_bs = np.zeros((bs, n_w))
    #         for i in range(bs):
    #             resample = np.random.randint(0, nbin, nbin)
    #             A_xx_bs[i] = A_xx
    #             re_sigmas_xx_bs[i] = re_sigmas_xx
    #     else:
    #         pass
    
    # def _get_sigma_xy_bins(self, resample, ws, dws, krnl, mdl, opt_method='cvxpy'):
    #     """Doesn't work yet, just copy-pasted but would be nice to have."""
    #     # Find sigma_xx
    #     krnl = maxent.kernel_b(beta, taus[0 : L // 2 + 1], ws, sym=True)
    #     re_sigmas_xx, A_xx = self._get_sigma_xx_bins(resample, ws, dws, krnl, mdl, opt_method)

    #     # Maxent sum
    #     krnl = maxent.kernel_b(beta, self.taus[:-1], ws, sym=False)
    #     f = np.append(self.chi_xx[resample].mean(0), self.chi_xx[resample].mean(0)[0]) - np.real(1j*np.append(self.chi_xy[resample].mean(0), -self.chi_xy[resample].mean(0)[0]))
    #     chiq0w0 = CubicSpline(taus, f).integrate(0, beta)
    #     g = (self.chi_xx[resample] - np.real(1j*self.chi_xy[resample])) / chiq0w0
    #     if opt_method == 'Bryan':
    #         A_sum = maxent.maxent(g, krnl, mdl, opt_method=opt_method, inspect=False)
    #     elif opt_method == 'cvxpy':
    #         N = len(ws)
    #         b = 2*A_xx[N//2:]
    #         B = np.hstack((np.flip(np.identity(N//2), axis=0), np.identity(N//2)))
    #         A_sum = maxent.maxent(g, krnl, mdl, opt_method=opt_method, constr_matrix=B, constr_vec=b, inspect=False)
    #     sigmas_sum = np.real(A_sum / dws * (chiq0w0 / self.sign[resample].mean())) * np.pi
    #     im_sigmas_xy_bs[i] = sigmas_sum-re_sigmas_xx
        
    #     ### Get re_sigmas_xy
    #     w_lim = np.max(w)
    #     fs = CubicSpline(w, im_sigmas_xy_bs[i])(xs)
    #     re_sigmas_xy_bs[i] = -np.imag(scipy.signal.hilbert(fs))

    # def plot_chi_xy(self):
    #     """Doesn't work yet, just copy-pasted but would be nice to have."""
    #     fig, ax = plt.subplots(figsize=(6, 2), ncols=3, layout='constrained')
    #     titles = [r"$ \langle j_x j_y \rangle $", r"$ \langle j_y j_x \rangle $", r"$\chi_{xy}$ (tot)"]
    #     ax[0].plot(np.mean(np.imag(jxjyq0), axis=0))
    #     ax[1].plot(np.mean(np.imag(jyjxq0), axis=0))
    #     ax[2].plot(np.mean(np.imag(chi_xy), axis=0))
    #     for i in range(3): 
    #         ax[i].set_title(titles[i])
    #         ax[i].set_xlabel(r'$\tau$')
    #     plt.suptitle(rf"U = {U}, $\beta$ = {beta}")




# class cond_calculator:
#     # Each of these objects calculates the cond. for a (U, beta) dataset, each with many bins/independently seeded MC runs
#     # For a given measured quantity (e.g. j-j correlator),
#     # util.load(dataset) returns an array of the summed measurement separated by bin
    
#     def __init__(self, path):
#         # Store simulation parameters
#         self.Ny, self.Nx, self.beta, self.L, self.tp = util.load_firstfile(
#             path, "metadata/Nx", "metadata/Ny", "metadata/beta", "params/L", "metadata/t'"
#         )
#         self.T = 1/self.beta
#         self.taus = np.linspace(0, beta, L + 1)
        
#         self.jj, self.sign, self.n_sample, self.n_bin = self._load_data() # note: sign and jj are already divided by n_sample
#         self.jjq0, self.chi_xx, self.chi_xy = self._prep_jjq0()
        
#     def _load_data(self):
#         # Load in all measurements
#         n_samples, sign, jj = util.load(
#             path, "meas_uneqlt/n_sample", "meas_uneqlt/sign", "meas_uneqlt/jj"
#         )
#         # Only keep measurements from bins where n_sample is max value (i.e., bin is complete)
#         n_sample = n_samples.max()   
#         mask = n_samples == n_sample
#         sign, jj = signs[mask], jjs[mask]
#         n_bin = mask.sum()
#         # Reshape jj into a 2x2 matrix (or 4x4, for nonzero t') discretized in tau + containing info from bond to bond (indexed by lattice site)
#         jj.shape = -1, L, 2, 2, Ny, Nx
#         # I guess also note that jj does not include tau=beta, remember len(taus) != L
#         return jj/n_sample, sign/n_sample, n_sample, n_bin

#     def _prep_jjq0(self):
#         """Get jjq0 and averaged xx and xy correlators."""
#         # Will probably need to change, this only works for tp=0
#         jjq0 = self.jj.sum((-1, -2))   # sum over all the bonds (since q=0)
#         # Get average longitudinal jj
#         jxjxq0 = -jjq0[..., 0, 0]
#         jyjyq0 = -jjq0[..., 1, 1]
#         chi_xx = 0.5 * (jxjxq0 + jyjyq0) # average over xx and yy to get avg longitudinal j-j
#         chi_xx = 0.5 * (chi_xx + chi_xx[:, -np.arange(L) % L]) # symmetrize bin by bin
#         chi_xx = np.real(chi_xx) # added for nflux != 0 data, should be purely real
#         # Get average transverse jj
#         jxjyq0 = jjq0[..., 1, 0]   # ***is this supposed to be minus or plus? idk but clearly one of them should be minus
#         jyjxq0 = -jjq0[..., 0, 1]
#         chi_xy = 0.5*(jxjyq0+jyjxq0)
#         chi_xy = 0.5 * (chi_xy - chi_xy[:, -np.arange(L) % L])
#         chi_xy = np.concatenate((np.expand_dims(chi_xy[:, 0], axis=1), 0.5*(chi_xy[:, 1:] - chi_xy[:, :0:-1])), axis=1) # antisymmetrize bin by bin
#         chi_xy = 1j*np.imag(chi_xy) # added for nflux != 0 data, should be purely imaginary
#         return jjq0, chi_xx, chi_xy
        
#     def get_sigma_xx(self, ws, dws, mdl, opt_method='Bryan', bs=0):
#         n_w = len(ws)
#         run_params = {
#                 'sigma_type': 'xx',
#                 'bs': bs,
#                 'opt_method': opt_method,
#                 'symm_ker': symm
#             )
#         krnl = maxent.kernel_b(beta, taus[0 : L // 2 + 1], ws, sym=True)
#         if mdl is None:
#             mdl = maxent.model_flat(dws)
            
#         if bs:
#             A_xx_bs = np.zeros((bs, n_w))
#             re_sigmas_xx_bs = np.zeros((bs, n_w))
#             for i in range(bs):
#                 resample = np.random.randint(0, self.nbin, self.nbin)
#                 re_sigmas_xx_bs[i], A_xx_bs[i] = self._get_sigma_xx_bins(resample, ws, dws, krnl, mdl, opt_method, symm=symm)
#             # re_sigmas_xx_mean = np.mean(re_sigmas_xx_bs, axis=0) 
#             # re_sigmas_xx_std = np.std(re_sigmas_xx_bs, axis=0)
#             results = {
#                 're_sigmas_xx_bs': re_sigmas_xx_bs,
#                 'A_xx_bs': A_xx_bs
#             }
#         else:
#             all_bins = np.arange(self.n_bin)
#             re_sigmas_xx, A_xx = self._get_sigma_xx_bins(all_bins, ws, dws, krnl, mdl, opt_method, symm=symm)
#             results = {
#                 're_sigmas_xx': re_sigmas_xx,
#                 'A_xx': A_xx
#             }
#         # Create and return a sigma object
#         sigmas_xx = sigma(self, ws, dws, run_params, results)
    
#     def _get_sigma_xx_bins(self, resample, ws, dws, krnl, mdl, opt_method):
#         f = self.chi_xx[resample].mean(0)
#         chiq0w0 = CubicSpline(self.taus, np.append(f, f[0])).integrate(0, beta)
#         g = 2 * chi_xx[resample, : self.L // 2 + 1] / chiq0w0
#         A_xx = maxent.maxent(g, krnl, mdl, opt_method=opt_method)
#         re_sigmas_xx = A_xx / dws * (chiq0w0 / sign[resample].mean()) * np.pi / 2*2
#         return A_xx, re_sigmas_xx

#     def get_sigma_xy(self, ws, dws, mdl, opt_method='Bryan', symm=True, bs=0):
#         run_params = {
#                 'sigma_type': 'xx',
#                 'bs': bs,
#                 'opt_method': opt_method,
#                 'symm_ker': symm
#         }
#         # Maybe we'll handle opt_method and mdl just like before, as a list
#         krnl = maxent.kernel_b(beta, taus[:-1], ws, sym=False)
#         if mdl is None:
#             mdl = maxent.model_flat(dws)
            
#         if bs:
#             A_xx_bs = np.zeros((bs, n_w))
#             re_sigmas_xx_bs = np.zeros((bs, n_w))
#             for i in range(bs):
#                 resample = np.random.randint(0, nbin, nbin)
#                 A_xx_bs[i] = A_xx
#                 re_sigmas_xx_bs[i] = re_sigmas_xx
#         else:
#             pass
    
#     def _get_sigma_xy_bins(self, resample, ws, dws, krnl, mdl, opt_method='cvxpy'):
#         """Doesn't work yet, just copy-pasted but would be nice to have."""
#         # Find sigma_xx
#         krnl = maxent.kernel_b(beta, taus[0 : L // 2 + 1], ws, sym=True)
#         re_sigmas_xx, A_xx = self._get_sigma_xx_bins(resample, ws, dws, krnl, mdl, opt_method)

#         # Maxent sum
#         krnl = maxent.kernel_b(beta, self.taus[:-1], ws, sym=False)
#         f = np.append(self.chi_xx[resample].mean(0), self.chi_xx[resample].mean(0)[0]) - np.real(1j*np.append(self.chi_xy[resample].mean(0), -self.chi_xy[resample].mean(0)[0]))
#         chiq0w0 = CubicSpline(taus, f).integrate(0, beta)
#         g = (self.chi_xx[resample] - np.real(1j*self.chi_xy[resample])) / chiq0w0
#         if opt_method == 'Bryan':
#             A_sum = maxent.maxent(g, krnl, mdl, opt_method=opt_method, inspect=False)
#         elif opt_method == 'cvxpy':
#             N = len(ws)
#             b = 2*A_xx[N//2:]
#             B = np.hstack((np.flip(np.identity(N//2), axis=0), np.identity(N//2)))
#             A_sum = maxent.maxent(g, krnl, mdl, opt_method=opt_method, constr_matrix=B, constr_vec=b, inspect=False)
#         sigmas_sum = np.real(A_sum / dws * (chiq0w0 / self.sign[resample].mean())) * np.pi
#         im_sigmas_xy_bs[i] = sigmas_sum-re_sigmas_xx
        
#         ### Get re_sigmas_xy
#         w_lim = np.max(w)
#         fs = CubicSpline(w, im_sigmas_xy_bs[i])(xs)
#         re_sigmas_xy_bs[i] = -np.imag(scipy.signal.hilbert(fs))

#     def plot_chi_xy(self):
#         """Doesn't work yet, just copy-pasted but would be nice to have."""
#         fig, ax = plt.subplots(figsize=(6, 2), ncols=3, layout='constrained')
#         titles = [r"$ \langle j_x j_y \rangle $", r"$ \langle j_y j_x \rangle $", r"$\chi_{xy}$ (tot)"]
#         ax[0].plot(np.mean(np.imag(jxjyq0), axis=0))
#         ax[1].plot(np.mean(np.imag(jyjxq0), axis=0))
#         ax[2].plot(np.mean(np.imag(chi_xy), axis=0))
#         for i in range(3): 
#             ax[i].set_title(titles[i])
#             ax[i].set_xlabel(r'$\tau$')
#         plt.suptitle(rf"U = {U}, $\beta$ = {beta}")

# class sigma():
#     """Represents a specific sigma calculation."""
#     def __init__(self, cond_calculator, sigma_type, ws, dws, run_params, results):
#         # This is more or less just a data storage object
#         # Access the dataset and parameters from the cond_calculator
#         self.cond_calculator = cond_calculator
#         self.U = cond_calculator.U
#         self.beta = cond_calculator.beta
#         self.T = cond_calculator.T

#         self.ws = ws
#         self.dws = dws
#         self.n_w = len(ws)
#         # Maybe I just pass it a dictionary of data/params?
        
#         # What do I need to know? cond_calculator stuff: U, beta
#         # ws, dws
#         # the actual spectra: A and sigmas, plus bootstrapped arrays, means and errors, etc.
#         # How was it run? if xx, need to know opt_method, whether symm kernel used, whether bootstrapped or not
#         # if xy, need to know opt_method, whether bootstrapped

#     def plot():
#         pass


# Maybe it makes more sense for every run/conductivity to be its own object?
# with plot_sigma, sigma_type, ws, alpha, etc. attached
# what about bootstraps
# idk but get_sigma returns a conductivity object, and also sets it as an attribute?
# I like the modularity of not having to calculate sigma_xy if you don't need it
# so maybe sigma_xy calls sigma_xx if it doesn't have it yet or something like that
# but then what about
# calculation of sigma moved to the sigma object
# I create a sigma object of some specified type by feeding it ws, dws, opt_method, etc.
# each sigma object has associated
# run parameters plus bootstrapping array, bootstrapping mean array (which I guess is the actual measurement,
# bs=0 array, etc.
# maybe make subclasses? lmfao
# or instead, maybe I'll just make a separate cond_calculator for diff optimizations types? I can always change this but I think this is simplest for now
        
def get_sigmas(path, w, dw, get_xy, bs=0, mdl=None, opt_method='Bryan', plot_chi_xy=False):
    """ Get sigma_xx and sigma_xy with MaxEnt.
        Args:
            path: filepath to folder containing bins of certain U, beta
            w: list of frequencies
            dw: list of frequency spacings
            get_xy: whether to calculate sigma_xy
            mdl: same deal as with opt_method
            opt_method: optimization method(s) to use in MaxEnt, if len 2 list then method for xx and sum, otherwise same method used for both
            plot_chi_xy: whether to plot chi_xy(tau) data
        Returns:
            re_sigma_xx
            if get_xy: re_sigma_xy, im_sigma_xy, sigmas_sum
    """
    ### Prep current-current data
    # Get model params
    Ny, Nx, beta, L, tp = util.load_firstfile(
        path, "metadata/Nx", "metadata/Ny", "metadata/beta", "params/L", "metadata/t'"
    )
    # Load in measurements
    n_sample, sign, jj = util.load(
        path, "meas_uneqlt/n_sample", "meas_uneqlt/sign", "meas_uneqlt/jj"
    )
    taus = np.linspace(0, beta, L + 1)
    # Only keep measurements from bins where n_sample is max value
    mask = n_sample == n_sample.max()   
    sign, jj = sign[mask], jj[mask]
    nbin = mask.sum()
    # Reshape jj into a 2x2 matrix (or 4x4, for nonzero t') discretized in tau + containing info from bond to bond (indexed by lattice site)
    jj.shape = -1, L, 2, 2, Ny, Nx
    jjq0 = jj.sum((-1, -2))   # sum over all the bonds (since q=0)

    ### Calculate re_sigmas_xx
    # Get average longitudinal jj
    jxjxq0 = -jjq0[..., 0, 0]
    jyjyq0 = -jjq0[..., 1, 1]
    chi_xx = 0.5 * (jxjxq0 + jyjyq0) # average over xx and yy to get avg longitudinal current current data (improved statistics)

    plt.scatter(taus[:-1], np.mean(chi_xx, axis=0))
    print(np.mean(0.5 * (chi_xx + chi_xx[::-1]), axis=0))
    chi_xx = 0.5 * (chi_xx + chi_xx[:, -np.arange(L) % L]) # symmetrize the data bin by bin
    print(np.mean(chi_xx, axis=0))
    chi_xx = np.real(chi_xx) # added for nflux != 0 data, chiq0 should be purely real, (nbin, ntau)

    # MaxEnt
    print("MaxEnt for sigma_xx")
    chi_xx /= n_sample.max()   
    sign /= n_sample.max()   # get avg measurements for each bin
    # Choose kernel and model to use (bosonic symmetric for sigma_xx)
    krnl = maxent.kernel_b(beta, taus[0 : L // 2 + 1], w, sym=True)
    ###***took out useBT stuff, maybe add back in at some point
    if mdl is None:
        mdl = maxent.model_flat(dw)
    if bs==0:
        f = chi_xx.mean(0)   # mean among bins, separated by tau, shape (10)
        chiq0w0 = CubicSpline(taus, np.append(f, f[0])).integrate(0, beta)  # G^O(i w_n = 0), used in def of G and A
        g = 2 * chi_xx[:, : L // 2 + 1] / chiq0w0
        opt_method_xx = opt_method if isinstance(opt_method, str) else opt_method[0]
        A_xx = maxent.maxent(g, krnl, mdl, opt_method=opt_method_xx)
        ####### ok this is a temporary step (to improve alpha selection)
        A_xx = 0.5*(A_xx+A_xx[::-1])
        re_sigmas_xx = np.real(A_xx / dw * (chiq0w0 / sign.mean())) * np.pi/2*2 #### final factor of 2 for range of w centered at (not starting at) 0
    elif bs != 0 and not get_xy:
        A_xx_bs = np.zeros((bs, len(w)))
        re_sigmas_xx_bs = np.zeros((bs, len(w)))
        for i in range(bs):
            resample = np.random.randint(0, nbin, nbin)
            f = chi_xx[resample].mean(0)
            chiq0w0 = CubicSpline(taus, np.append(f, f[0])).integrate(0, beta)
            g = 2 * chi_xx[resample, : L // 2 + 1] / chiq0w0
            opt_method_xx = opt_method if isinstance(opt_method, str) else opt_method[0]
            A_xx = maxent.maxent(g, krnl, mdl, opt_method=opt_method_xx)
            A_xx_bs[i] = A_xx
            re_sigmas_xx_bs[i] = A_xx / dw * (chiq0w0 / sign[resample].mean()) * np.pi / 2*2

    if get_xy:
        # Get average transverse jj
        # current configuration: xy current correlator slopes down
        jxjyq0 = jjq0[..., 1, 0]   # ***is this supposed to be minus or plus? idk but clearly one of them should be minus
        jyjxq0 = -jjq0[..., 0, 1]
        chi_xy = 0.5*(jxjyq0+jyjxq0)
        plt.figure()
        plt.scatter(taus[:-1], np.imag(np.mean(chi_xy, axis=0)))
        chi_xy = np.concatenate((np.expand_dims(chi_xy[:, 0], axis=1), 0.5*(chi_xy[:, 1:] - chi_xy[:, :0:-1])), axis=1) ### antisymmetrizing step
        chi_xy = 1j*np.imag(chi_xy) # added for nflux != 0 data, should be purely imaginary

        # Optionally plot chi_xy(tau)
        if plot_chi_xy:
            fig, ax = plt.subplots(figsize=(6, 2), ncols=3, layout='constrained')
            titles = [r"$ \langle j_x j_y \rangle $", r"$ \langle j_y j_x \rangle $", r"$\chi_{xy}$ (tot)"]
            ax[0].plot(np.mean(np.imag(jxjyq0), axis=0))
            ax[1].plot(np.mean(np.imag(jyjxq0), axis=0))
            ax[2].plot(np.mean(np.imag(chi_xy), axis=0))
            for i in range(3): 
                ax[i].set_title(titles[i])
                ax[i].set_xlabel(r'$\tau$')
            plt.suptitle(rf"U = {U}, $\beta$ = {beta}")

        ### Get sigmas_sum with MaxEnt
        print("MaxEnt for sigmas_sum")
        chi_xy /= n_sample.max()    
        # Choose kernel and model to use (bosonic non-symmetric)
        krnl = maxent.kernel_b(beta, taus[:-1], w, sym=False)
        if bs == 0:
            f = np.append(chi_xx.mean(0), chi_xx.mean(0)[0]) - np.real(1j*np.append(chi_xy.mean(0), -chi_xy.mean(0)[0])) # clunky expression bc f no longer symmetric, but have to include tau = beta point
            chiq0w0 = CubicSpline(taus, f).integrate(0, beta)  # G^O(i w_n = 0), used in def of G and A
            g = (chi_xx - np.real(1j*chi_xy)) / chiq0w0
            opt_method_xy = opt_method if isinstance(opt_method, str) else opt_method[-1]
            
            N = krnl.shape[1]
            b = 2*A_xx[N//2:]
            B = np.hstack((np.flip(np.identity(N//2), axis=0), np.identity(N//2)))
            A = maxent.maxent(g, krnl, mdl, opt_method=opt_method_xy, constr_matrix=B, constr_vec=b, inspect=True)
            sigmas_sum = np.real(A / dw * (chiq0w0 / sign.mean())) * np.pi
            im_sigmas_xy = sigmas_sum-re_sigmas_xx
            
            ### Get re_sigmas_xy
            w_lim = np.max(w)
            xs = np.linspace(-w_lim, w_lim, 1500)
            fs = CubicSpline(w, im_sigmas_xy)(xs)
            re_sigmas_xy = -np.imag(scipy.signal.hilbert(fs))
        else:
            w_lim = np.max(w)
            xs = np.linspace(-w_lim, w_lim, 1500)
            re_sigmas_xx_bs = np.zeros((bs, len(w)))
            re_sigmas_xy_bs = np.zeros((bs, len(xs)))
            im_sigmas_xy_bs = np.zeros((bs, len(w)))
            
            N = krnl.shape[1]
            
            for i in range(bs):
                print(f"Bs {i} ------------")
                resample = np.random.randint(0, nbin, nbin)

                ### wtf I guess I need to recalculate sigma_xx for each resample too lmfao
                krnl = maxent.kernel_b(beta, taus[0 : L // 2 + 1], w, sym=True)
                f = chi_xx[resample].mean(0)
                chiq0w0 = CubicSpline(taus, np.append(f, f[0])).integrate(0, beta)
                g = 2 * chi_xx[resample, : L // 2 + 1] / chiq0w0
                opt_method_xx = opt_method if isinstance(opt_method, str) else opt_method[0]
                A_xx = maxent.maxent(g, krnl, mdl, opt_method=opt_method_xx)
                ####### ok this is a temporary step (to improve alpha selection)
                A_xx = 0.5*(A_xx+A_xx[::-1])   # this shouldn't really matter bc A_xx is usually super symmetric anyway
                re_sigmas_xx = A_xx / dw * (chiq0w0 / sign[resample].mean()) * np.pi / 2*2
                re_sigmas_xx_bs[i] = re_sigmas_xx

                krnl = maxent.kernel_b(beta, taus[:-1], w, sym=False)
                f = np.append(chi_xx[resample].mean(0), chi_xx[resample].mean(0)[0]) - np.real(1j*np.append(chi_xy[resample].mean(0), -chi_xy[resample].mean(0)[0]))
                chiq0w0 = CubicSpline(taus, f).integrate(0, beta)
                g = (chi_xx[resample] - np.real(1j*chi_xy[resample])) / chiq0w0
                opt_method_xy = opt_method if isinstance(opt_method, str) else opt_method[-1]
                b = 2*A_xx[N//2:]
                B = np.hstack((np.flip(np.identity(N//2), axis=0), np.identity(N//2)))
                A_xy = maxent.maxent(g, krnl, mdl, opt_method=opt_method_xy, constr_matrix=B, constr_vec=b, inspect=True if i%4==0 else False)
                sigmas_sum = np.real(A_xy / dw * (chiq0w0 / sign.mean())) * np.pi
                im_sigmas_xy_bs[i] = sigmas_sum-re_sigmas_xx
                ### Get re_sigmas_xy
                w_lim = np.max(w)
                fs = CubicSpline(w, im_sigmas_xy_bs[i])(xs)
                re_sigmas_xy_bs[i] = -np.imag(scipy.signal.hilbert(fs))
    if bs != 0:
        if get_xy:
            return re_sigmas_xx_bs, re_sigmas_xy_bs, im_sigmas_xy_bs
        else:
            return re_sigmas_xx_bs
    else:
        if get_xy:
            return re_sigmas_xx, re_sigmas_xy, im_sigmas_xy, sigmas_sum
        else:
            return re_sigmas_xx

def calc_sigmas(path, w, dw, get_xy, bs=0, mdl=None, opt_method='Bryan', plot_chi_xy=False):
    """ Get sigma_xx and (optionally) sigma_xy with MaxEnt.
        Args:
            path: filepath to folder containing bins of certain U, beta
            w: list of frequencies
            dw: list of frequency spacings
            get_xy: whether to calculate sigma_xy
            mdl: same deal as with opt_method
            opt_method: optimization method(s) to use in MaxEnt, if len 2 list then method for xx and sum, otherwise same method used for both
            plot_chi_xy: whether to plot chi_xy(tau) data
        Returns:
            re_sigma_xx
            if get_xy: re_sigma_xy, im_sigma_xy, sigmas_sum
    """
    # -------------------- Load and prep current-current correlator data --------------------
    # Get model params
    Ny, Nx, beta, L, tp = util.load_firstfile(
        path, "metadata/Nx", "metadata/Ny", "metadata/beta", "params/L", "metadata/t'"
    )
    # Load in measurements
    n_sample, sign, jj = util.load(
        path, "meas_uneqlt/n_sample", "meas_uneqlt/sign", "meas_uneqlt/jj"
    )
    taus = np.linspace(0, beta, L + 1)
    # Only keep measurements from bins where n_sample is max value
    mask = n_sample == n_sample.max()   
    sign, jj = sign[mask], jj[mask]
    nbin = mask.sum()
    # Reshape jj into a 2x2 matrix (or 4x4, for nonzero t') discretized in tau + containing info from bond to bond (indexed by lattice site)
    jj.shape = -1, L, 2, 2, Ny, Nx
    
    jjq0 = jj.sum((-1, -2))   # sum over all the bonds (since q=0)

    # -------------------- Calculate re_sigmas_xx --------------------
    

    if get_xy:
        # Get average transverse jj
        # current configuration: xy current correlator slopes down
        jxjyq0 = jjq0[..., 1, 0]   # ***is this supposed to be minus or plus? idk but clearly one of them should be minus
        jyjxq0 = -jjq0[..., 0, 1]
        chi_xy = 0.5*(jxjyq0+jyjxq0)
        chi_xy = np.concatenate((np.expand_dims(chi_xy[:, 0], axis=1), 0.5*(chi_xy[:, 1:] - chi_xy[:, :0:-1])), axis=1) ### antisymmetrizing step
        chi_xy = 1j*np.imag(chi_xy) # added for nflux != 0 data, should be purely imaginary

        # Optionally plot chi_xy(tau)
        if plot_chi_xy:
            fig, ax = plt.subplots(figsize=(6, 2), ncols=3, layout='constrained')
            titles = [r"$ \langle j_x j_y \rangle $", r"$ \langle j_y j_x \rangle $", r"$\chi_{xy}$ (tot)"]
            ax[0].plot(np.mean(np.imag(jxjyq0), axis=0))
            ax[1].plot(np.mean(np.imag(jyjxq0), axis=0))
            ax[2].plot(np.mean(np.imag(chi_xy), axis=0))
            for i in range(3): 
                ax[i].set_title(titles[i])
                ax[i].set_xlabel(r'$\tau$')
            plt.suptitle(rf"U = {U}, $\beta$ = {beta}")

        ### Get sigmas_sum with MaxEnt
        print("MaxEnt for sigmas_sum")
        chi_xy /= n_sample.max()    
        # Choose kernel and model to use (bosonic non-symmetric)
        krnl = maxent.kernel_b(beta, taus[:-1], w, sym=False)
        if bs == 0:
            f = np.append(chi_xx.mean(0), chi_xx.mean(0)[0]) - np.real(1j*np.append(chi_xy.mean(0), -chi_xy.mean(0)[0])) # clunky expression bc f no longer symmetric, but have to include tau = beta point
            chiq0w0 = CubicSpline(taus, f).integrate(0, beta)  # G^O(i w_n = 0), used in def of G and A
            g = (chi_xx - np.real(1j*chi_xy)) / chiq0w0
            opt_method_xy = opt_method if isinstance(opt_method, str) else opt_method[-1]
            
            N = krnl.shape[1]
            b = 2*A_xx[N//2:]
            B = np.hstack((np.flip(np.identity(N//2), axis=0), np.identity(N//2)))
            A = maxent.maxent(g, krnl, mdl, opt_method=opt_method_xy, constr_matrix=B, constr_vec=b, inspect=True)
            sigmas_sum = np.real(A / dw * (chiq0w0 / sign.mean())) * np.pi
            im_sigmas_xy = sigmas_sum-re_sigmas_xx
            
            ### Get re_sigmas_xy
            w_lim = np.max(w)
            xs = np.linspace(-w_lim, w_lim, 1500)
            fs = CubicSpline(w, im_sigmas_xy)(xs)
            re_sigmas_xy = -np.imag(scipy.signal.hilbert(fs))
        else:
            # get errors
            pass
    if get_xy:
        return re_sigmas_xx, re_sigmas_xy, im_sigmas_xy, sigmas_sum
    return re_sigmas_xx

def calc_sigmas_xx(jjq0, w, dw, get_xy, bs=0, mdl=None, opt_method='Bryan'):
    """Calculate re_sigmas_xx from (q=0) current-current correlator data."""
    # Get average longitudinal jj
    jxjxq0 = -jjq0[..., 0, 0]
    jyjyq0 = -jjq0[..., 1, 1]
    chi_xx = 0.5 * (jxjxq0 + jyjyq0) # average over xx and yy to get avg longitudinal current current data (improved statistics)
    chi_xx = 0.5 * (chi_xx + chi_xx[:, -np.arange(L) % L]) # symmetrize the data bin by bin
    chi_xx = np.real(chi_xx) # added for nflux != 0 data, chiq0 should be purely real
    # MaxEnt
    print("MaxEnt for sigma_xx")
    chi_xx /= n_sample.max()   
    sign /= n_sample.max()   # get avg measurements for each bin
    # Choose kernel and model to use (bosonic symmetric for sigma_xx)
    krnl = maxent.kernel_b(beta, taus[0 : L // 2 + 1], w, sym=True)
    ###***took out useBT stuff, maybe add back in at some point
    if mdl is None:
        mdl = maxent.model_flat(dw)
    if bs==0:
        f = chi_xx.mean(0)   # mean among bins, separated by tau, shape (10)
        chiq0w0 = CubicSpline(taus, np.append(f, f[0])).integrate(0, beta)  # G^O(i w_n = 0), used in def of G and A
        g = 2 * chi_xx[:, : L // 2 + 1] / chiq0w0
        opt_method_xx = opt_method if isinstance(opt_method, str) else opt_method[0]
        A_xx = maxent.maxent(g, krnl, mdl, opt_method=opt_method_xx)
        ####### ok this is a temporary step (to improve alpha selection)
        A_xx = 0.5*(A_xx+A_xx[::-1])
        re_sigmas_xx = np.real(A_xx / dw * (chiq0w0 / sign.mean())) * np.pi/2*2 #### final factor of 2 for range of w centered at (not starting at) 0
    else:
        # Calculate error with bootstrapping
        pass
    return re_sigmas_xx, A_xx

def plot_sigmas(ws, sigmas_w, U, beta, sigma_name, yerr=None, label='', ylim=None, xlim=None):
    # (for ylabel)
    sigma_name_dict = {
        "re_sigmas_xx": r'Re[$\sigma_{xx}(\omega)$]', 
        "im_sigmas_xx": r'Im[$\sigma_{xx}(\omega)$]',
        "re_sigmas_xy": r'Re[$\sigma_{xy}(\omega)$]',
        "im_sigmas_xy": r'Im[$\sigma_{xy}(\omega)$]',
        "sigmas_sum": r'Re[$\sigma_{xx}(\omega)$] + Im[$\sigma_{xy}(\omega)$]'
    }
    
    fig, ax = plt.subplots()
    # plt.errorbar(ws, re_sigmas_xx, yerr=re_sigmas_xx_err)
    if np.any(yerr):
        ax.errorbar(ws, sigmas_w, yerr=yerr, fmt='s-', lw=0.7, ms=0, label=label, capsize=0, ecolor='orange', elinewidth=0.5)
    else:
        ax.scatter(ws, sigmas_w, s=1, label=label)
    plt.xlabel(r'$\omega$')
    plt.ylabel(sigma_name_dict[sigma_name])
    plt.title(rf'{sigma_name_dict[sigma_name]}: U = {U}, $\beta$ = {beta}')
    if np.any(ylim):
        plt.ylim(*ylim)
    if np.any(xlim):
        plt.xlim(*xlim)
    if label:
        plt.legend()

def plot_sigmas_bs(ws, sigmas_bs, U, beta, sigma_name, yerr=None, label='', ylim=None, xlim=None):
    '''This version just plots all the bootstrapped sigmas on top of each other'''
    sigma_name_dict = {
        "re_sigmas_xx": r'Re[$\sigma_{xx}(\omega)$]', 
        "im_sigmas_xx": r'Im[$\sigma_{xx}(\omega)$]',
        "re_sigmas_xy": r'Re[$\sigma_{xy}(\omega)$]',
        "im_sigmas_xy": r'Im[$\sigma_{xy}(\omega)$]',
        "sigmas_sum": r'Re[$\sigma_{xx}(\omega)$] + Im[$\sigma_{xy}(\omega)$]'
    }
    num_bs = len(sigmas_bs)
    fig, ax = plt.subplots()
    for i in range(num_bs):
        ax.plot(ws, sigmas_bs[i], lw=1, color='#0C5DA5', alpha=0.9)
    plt.xlabel(r'$\omega$')
    plt.ylabel(sigma_name_dict[sigma_name])
    plt.title(rf'{sigma_name_dict[sigma_name]}: U = {U}, $\beta$ = {beta}, bs = {num_bs}')

# def plot_sigmas2(ws, bs_sigmas_w, U, beta, ylabel=''):
    # '''This version just plots all the bootstrapped sigmas on top of each other'''
    # num_bs = len(bs_sigmas_w)
    # fig, ax = plt.subplots(figsize = [6.4, 4.8])
    # for i in range(num_bs):
    #     ax.plot(ws, bs_sigmas_w[i], lw=1, color='#0C5DA5', alpha=0.9)
    # plt.xlabel(r'$\omega$')
    # plt.ylabel(ylabel)
    # plt.title(rf'{ylabel}: U = {U}, $\beta$ = {beta}')