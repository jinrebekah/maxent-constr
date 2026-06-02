import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import optical_cond as oc
from tqdm import tqdm
import seaborn as sns
import scipy

from uncertainties import ufloat
from uncertainties import unumpy as unp
import sys
import os
if os.path.exists('/oak/stanford/orgs/simes/rebjin/dqmc-dev/util'):
    sys.path.append('/oak/stanford/orgs/simes/rebjin/dqmc-dev/util')
else:
    sys.path.append('/Users/rebekahjin/Documents/devereaux_group/dqmc-dev/util')
# sys.path.append('/oak/stanford/orgs/simes/rebjin/dqmc-dev/util')
import util

import scienceplots
plt.style.use(['science','no-latex'])
# plt.style.use(['science'])
# plt.rcParams['text.usetex'] = True
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats("svg") 
default_figsize = plt.rcParams['figure.figsize']

plt.rcParams.update({
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 12,
    'legend.labelspacing': 0.35,
    'lines.linewidth': 1.2
})

ERRORBAR_STYLE = dict(
    markersize=2,
    elinewidth=1,
    capsize=0.8,
)


### functions to load sigma data into dfs
def calc_rho_df(pickle_folder, nfluxes, ns, Us, betas, mus=[None], use_xy=True, calc_xx_proxy=False, calc_xy_proxy=False, save_csv=False):
    """Calculates rhos for range of Us, ns, Bs, betas, and returns df.
    Assumes a particular pickle folder path convention, may need to modify for others.
    """
    # example pickle_folder = '/scratch/users/rebjin/kohler_pickles/8x8_tp0/'
    row_dicts = []
    with tqdm(total=len(Us)*len(ns)*len(betas)*(len(nfluxes))) as pbar:
        for U in Us:
            for n in ns:
                for beta in betas:
                    for nflux in nfluxes:
                        for mu in mus:
                            try:
                                sig = oc.get_sig_pickle(pickle_folder, nflux=nflux, n=n, U=U, beta=beta, mu=mu)
                            
                                # Calculate DC rho_xx
                                dict = {'Nx': int(sig.Nx), 'Ny': int(sig.Ny), 'tp': float(sig.tp), 'U': U, 'n': n, 'beta': beta, 'nflux': nflux}
                                if mu is not None: dict.update({'mu': mu})
                                rho, rho_err, sig_xx, sig_xx_err, sig_xy, sig_xy_err = oc.calc_rho_xx_0(sig, use_xy=use_xy)
                                dict.update({'rho_xx': rho, 'rho_xx_err': rho_err, 'sig_xx': sig_xx, 'sig_xx_err': sig_xx_err, 'sig_xy': sig_xy, 'sig_xy_err': sig_xy_err})
    
                                # Optionally calc proxies
                                if calc_xx_proxy:
                                    proxy1, proxy1_err, proxy2, proxy2_err = oc.calc_rho_proxy(sig)
                                    dict.update({'rho_proxy1':proxy1, 'rho_proxy1_err':proxy1_err, 'rho_proxy2':proxy2, 'rho_proxy2_err':proxy2_err})
                                    dict.update({'sig_xx_proxy1':1/proxy1, 'sig_xx_proxy1_err':proxy1_err/proxy1**2, 'sig_xx_proxy2':1/proxy2, 'sig_xx_proxy2_err':proxy2_err/proxy2**2})
                                if calc_xy_proxy:
                                    proxy_xy, proxy_xy_err = oc.calc_sigma_xy_proxy(sig)
                                    dict.update({'sig_xy_proxy':proxy_xy, 'sig_xy_proxy_err':proxy_xy_err})
    
                                # # maybe temporary but also add the actual n measurement
                                n_samples, sign, density = util.load(sig.path, "meas_uneqlt/n_sample", "meas_eqlt/sign", "meas_eqlt/density")
                                mask = (n_samples == n_samples.max())
                                sign, density = sign[mask], density[mask]
                                nj = util.jackknife(sign, density.sum(1))
                                dict.update({'n_meas': nj[0], 'n_meas_err': nj[1]})
    
                                # calculate double_occ (not sure double check)
                                double_occ = util.load(sig.path, "meas_eqlt/double_occ")
                                double_occ = np.reshape(double_occ, np.shape(density))                
                                double_occ = double_occ[mask]
                                result = util.jackknife(sign, double_occ.sum(1))
                                dict.update({'double_occ': result[0], 'double_occ_err': result[1]})
                                
                                # Save in df
                                row_dicts.append(dict)
                            # Update progress bar
                            except Exception as e:
                                print('Error: ', e)
                            finally:
                                pbar.update(1)
    df = pd.DataFrame(row_dicts)
    return df

def calc_MR_df(rho_df, B0=0, calc_xx_proxy=False, save_csv=False):
    """Calculates MRs from df of rhos and returns same df + MR data.
    B0 is generally 0, the zero-field value to normalize MR with. May sometimes use 1 due to finite size issues.
    """    
    # example pickle_folder = '/scratch/users/rebjin/kohler_pickles/8x8_tp0/'
    
    def _calc_MR_group(group):
        # group is a df subset of rho_df separated by U, n, beta, with nflux
        # Also need to handle different B0
        
        # first need to check whether desired B0 exists -- if not, don't calculate anything for this group
        Nx, Ny, tp, U, n, beta = group.name
        
        group = group.sort_values(by='nflux', ascending=True)
        if B0 not in group['nflux'].values:
            print(f'No nflux={B0} data for U={U}, n={n}, beta={beta}')
            return pd.DataFrame()
        
        # now calculate MR and error
        group = group[group['nflux'] >= B0] # drop all rows of nflux < B0 (will only ever be 0 lol)
        rho_uc = unp.uarray(group['rho_xx'], group['rho_xx_err'])
        rho_B0_uc = rho_uc[0] # can do this bc group should be sorted
        MR_uc_arr = (rho_uc - rho_B0_uc) / rho_B0_uc
        group['MR'] = unp.nominal_values(MR_uc_arr)
        group['MR_err'] = unp.std_devs(MR_uc_arr)
        
        # optionally calc proxy "MRs" (kinda dumb but see what happens)
        proxy_names = ['rho_proxy1', 'rho_proxy2']
        if calc_xx_proxy:
            for proxy_name in proxy_names:
                if proxy_name in group.columns:
                    rho_proxy_uc = unp.uarray(group[proxy_name], group[proxy_name+'_err'])
                    rho_proxy_B0_uc = rho_proxy_uc[0]
                    MR_uc_arr = (rho_proxy_uc - rho_proxy_B0_uc) / rho_proxy_B0_uc
                    group['MR_'+proxy_name] = unp.nominal_values(MR_uc_arr)
                    group['MR_'+proxy_name+'_err'] = unp.std_devs(MR_uc_arr)
        # adjust nfluxes for when B0 > 0 used
        group['nflux'] = group['nflux']-B0
        return group

    # MR_df = rho_df.groupby(['U', 'n', 'beta'], as_index=True ).apply(_calc_MR_group)
    MR_df = rho_df.groupby(['Nx', 'Ny', 'tp', 'U', 'n', 'beta'], as_index=True ).apply(_calc_MR_group)
    MR_df = MR_df.reset_index(drop=True) # kept this bc it makes getting U, beta, n convenient
    return MR_df

def calc_MR(rho, rho_err, rho_B0, rho_B0_err):
    '''Just calculates magnetoresistance and error: MR(B)=rho(B)-rho_B0/rho_B0'''
    rho_uc = ufloat(rho, rho_err)
    rho_B0_uc = ufloat(rho_B0, rho_B0_err)
    MR_uc = ((rho_uc-rho_B0_uc)/rho_B0_uc)
    MR = MR_uc.nominal_value
    MR_err = MR_uc.std_dev
    # MR_err = np.sqrt((1/rho_B0*rho_err)**2 + (rho/rho_B0**2*rho_B0_err)**2) this is the correct formula
    
    return MR, MR_err

######################## plots vs. B
def _setup_plot(df, lines, x_label, y_label, Nx=None, Ny=None, tp=None, color_dict=None, ax=None, figsize=None, palette='husl'):
    # small helper bc this is repeated in every single function lol
    # lines is either betas or nfluxes
    Nx = _resolve_param(df, Nx, "Nx")
    Ny = _resolve_param(df, Ny, "Ny")
    tp = _resolve_param(df, tp, "tp")
    
    if color_dict is None:
        colors = sns.color_palette(palette, len(lines))
        color_dict = dict(zip(lines, colors))
    
    if figsize is None:
        figsize = (default_figsize[0], default_figsize[1]*1.5)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
        
    return Nx, Ny, tp, color_dict, ax

def _resolve_param(df, value, name):
    # tiny helper to require Nx, Ny, tp specified if df contains multiple values
    if value is not None:
        return value
    if df[name].nunique() > 1:
        raise ValueError(f"df contains multiple {name} values, must specify {name}.")
    return df[name].iloc[0]

def plot_dc_sig_B(df, nfluxes, n, U, betas, mode, Nx=None, Ny=None, tp=None, ylim=None, xlim=None, ax=None, show_legend=True, color_dict=None, style={}):
    if mode=='xx':
        sig_label = 'sig_xx'
        y_label = rf'$\sigma_{{xx}}$'
    elif mode=='xy':
        sig_label = 'sig_xy'
        y_label = rf'$\sigma_{{xy}}$'
    else:
        pass
    x_label = r'$B$ $[\Phi_0/a^2]$'

    Nx, Ny, tp, color_dict, ax = _setup_plot(df, betas, x_label, y_label, Nx=Nx, Ny=Ny, tp=tp, color_dict=color_dict, ax=ax)
    for k, beta in enumerate(betas):
        df_subset = df[(df['Nx']==Nx) & (df['Ny']==Ny) & (df['tp']==tp) & (df['U']==U) & (df['beta']==beta) & (df['n']==n) & (df['nflux'].isin(nfluxes))].sort_values(by='nflux').reset_index()
        if df_subset.empty:
            continue
        sigs = df_subset[sig_label]
        sig_errs = df_subset[sig_label+'_err']
        Bs_plot = df_subset['nflux']/(Nx*Ny)
        # ax.errorbar(Bs_plot, sigs, yerr=sig_errs, label=rf'$\beta$={beta}', marker='o', markersize=2, color = color_dict[beta], elinewidth=1, capsize=0.8, **style)
        ax.errorbar(Bs_plot, sigs, yerr=sig_errs, label=rf'$\beta$={beta}', color = color_dict[beta], **style)
        
    if show_legend: ax.legend()
    plt.tight_layout()

def plot_rho_B(df, nfluxes, ns, Us, betas,  Nx=None, Ny=None, tp=None, mode='rho', B0=0, ylims=None, xlim=None, suptitle='', ax=None, show_legend=True):
    # mode options: ['rho', 'rho_diff', 'proxy1', 'proxy2']
    if mode=='rho':
        rho_label='rho_xx'
        y_label = r'$\rho_{xx}(B)$'
    elif mode=='rho_diff':
        y_label = r'$\rho_{xx}(B)-\rho_{xx}(0)$'
    elif mode=='proxy1':
        rho_label='rho_proxy1'
        y_label = r'$\rho_{1}$'
    elif mode=='proxy2':
        rho_label='rho_proxy2'
        y_label = r'$\rho_{2}$'
    else:
        pass

    Nx = _resolve_param(df, Nx, "Nx")
    Ny = _resolve_param(df, Ny, "Ny")
    tp = _resolve_param(df, tp, "tp")
    
    colors = sns.color_palette('husl', len(betas))
    if len(Us)==1 and len(ns)==1:
        figsize = (default_figsize[0], default_figsize[1]*1.5)
    else:
        figsize = (default_figsize[0]*len(ns)*0.7, default_figsize[1]*1*len(Us))

    if ax is None:
        fig, ax = plt.subplots(nrows=len(Us), ncols = len(ns), figsize=figsize, sharey=True)
    ax = np.reshape(ax, (len(Us), len(ns)))
    
    for i, U in enumerate(Us):
        for j, n in enumerate(ns):
            for k, beta in enumerate(betas):
                df_subset = df[(df['Nx']==Nx) & (df['Ny']==Ny) & (df['tp']==tp) & (df['U']==U) & (df['beta']==beta) & (df['n']==n) & (df['nflux'].isin(nfluxes))].sort_values(by='nflux').reset_index()
                if df_subset.empty:
                    continue
                if mode=='rho_diff':
                    rho_0 = df_subset[(df_subset['nflux']==B0)][f'rho_xx'].item()
                    rho_0_err = df_subset[(df_subset['nflux']==B0)][f'rho_xx_err'].item()
                    rhos = df_subset[f'rho_xx'] - rho_0
                    rho_errs = np.sqrt((df_subset[f'rho_xx_err'])**2+(rho_0_err)**2)
                else:
                    rhos = df_subset[rho_label]
                    rho_errs = df_subset[rho_label+'_err']
                Bs_plot = df_subset['nflux']/(df_subset['Nx'][0]*df_subset['Ny'][0])
                lw = 1 if ":" in fmt else 0.75
                ax[i, j].errorbar(Bs_plot, rhos, yerr=rho_errs, label=rf'$\beta$={beta}', fmt=fmt, markersize=2, color = colors[betas.index(beta)], lw=lw, elinewidth=1, capsize=0.8)
            ax[i, j].tick_params(axis='both', which='major', labelsize=12)

            # Make background grey if no data
            if not (ax[i, j].lines or ax[i, j].collections):
                ax[i, j].set_facecolor('gainsboro')
                
    # Plot labels
    for j, n in enumerate(ns):
        ax[-1, j].set_xlabel(r'$B$ $[\Phi_0/a^2]$', fontsize=12)
        ax[0, j].set_title(f'n = {n}', fontsize=14)
    for i, U in enumerate(Us):
        ax[i, 0].set_ylabel(y_label, fontsize=12)
        ax[i, -1].annotate(f'U = {U}', (1.02, 0.48), rotation=270, xycoords='axes fraction', fontsize=14)
    if show_legend: ax[0,-1].legend(labelspacing = 0.35, fontsize=9)

    plt.tight_layout()

def plot_MR(df, nfluxes, ns, Us, betas, Nx=None, Ny=None, tp=None, mode='rho', shared_axes=True, ylim=None, xlim=None, suptitle='', calc_fit=False, loglog=False, fmt='-', ax=None, show_legend=True, show_title=True, labels=None, colors=None):
    # modes can be rho, proxy1, proxy2
    if mode=='rho':
        MR_label='MR'
        y_label = r'$MR$ [%]'
    elif mode=='proxy1':
        MR_label='MR_rho_proxy1'
        y_label = r'$MR_{1}$ [%]'
    elif mode=='proxy2':
        MR_label='MR_rho_proxy2'
        y_label = r'$MR_{2}$ [%]'
    else:
        pass

    Nx = _resolve_param(df, Nx, "Nx")
    Ny = _resolve_param(df, Ny, "Ny")
    tp = _resolve_param(df, tp, "tp")

    if colors is None:
        colors = sns.color_palette('husl', len(betas))
    if len(Us)==1 and len(ns)==1:
        figsize = (default_figsize[0], default_figsize[1]*1.5)
    else:
        figsize = (default_figsize[0]*len(ns)*0.7, default_figsize[1]*1.3*len(Us))
        # figsize = (default_figsize[0]*len(ns)*0.9, default_figsize[1]*1.2*len(Us))
    if ax is None:
        fig, ax = plt.subplots(nrows=len(Us), ncols = len(ns), figsize=figsize, sharey=True)
    ax = np.reshape(ax, (len(Us), len(ns)))

    if xlim is None:
        x_min = np.min(nfluxes)/(Nx*Ny)
        x_max = np.max(nfluxes)/(Nx*Ny)
        xlim = [x_min-(x_max-x_min)*0.1, x_max+(x_max-x_min)*0.1]
    
    for i, U in enumerate(Us):
        if calc_fit:
            print("Fitting MRs to power law A*x^B")
            print(f"{'n':>6}  {'U':>4}  {'beta':>4}  {'A':>10} ± {'dA':<10}  {'B':>10} ± {'dB':<10}")
            print("-" * 84)
        for j, n in enumerate(ns):
            for k, beta in enumerate(betas):
                # df_subset = df[(df['Nx']==Nx) & (df['Ny']==Ny) & (df['tp']==tp) & (df['U']==U) & (df['beta']==beta) & (df['n']==n) & (df['nflux'].isin(nfluxes)) & (df['nflux']/(Nx*Ny) < xlim[1])].sort_values(by='nflux').reset_index()
                df_subset = df[(df['Nx']==Nx) & (df['Ny']==Ny) & (df['tp']==tp) & (df['U']==U) & (df['beta']==beta) & (df['n']==n) & (df['nflux'].isin(nfluxes))].sort_values(by='nflux').reset_index()
                if df_subset.empty:
                    continue
                MRs = df_subset[MR_label]*100
                MR_errs = df_subset[MR_label+'_err']*100
                Bs_plot = df_subset['nflux']/(Nx*Ny)
                lw = 1 if ':' in fmt else 0.75
                (_, caps, bars) = ax[i, j].errorbar(Bs_plot, MRs, yerr=MR_errs, label=labels[k] if labels else rf'$\beta$={beta}', fmt=fmt, lw=lw, elinewidth=1, color = colors[k], capsize=0.8)
                if loglog:
                    ax[i, j].set_yscale('log', nonpositive='clip')
                    ax[i, j].set_xscale('log', nonpositive='clip')

                # report fit coefficients
                if calc_fit:
                    def fit_func(logx, logA, C):
                        return logA + C*logx
                    mask = Bs_plot > 0 # exclude (0,0) point
                    params, cov = scipy.optimize.curve_fit(fit_func, np.log(Bs_plot[mask]/(8*8)), np.log(MRs[mask]), p0=[1, 2], sigma=MR_errs[mask]/MRs[mask], absolute_sigma=True)
                    perr = np.sqrt(np.diag(cov))
                    
                    print(f"{n:>6}  {U:>4}  {beta:>4}  {np.exp(params[0]):10.2f} ± {np.exp(params[0])*perr[0]:<10.2f}  "
                    f"{params[1]:10.3f} ± {perr[1]:<10.3f}")
                
            # Remove x and y axis labels from interior axes
            if shared_axes:
                if j!=0: ax[i, j].tick_params(labelleft=False)
                if i!=ax.shape[0]-1: ax[i, j].tick_params(labelbottom=False)
            if ylim is not None:
                ax[i, j].set_ylim(*ylim)
            if xlim is not None:
                ax[i, j].set_xlim(*xlim)
                
            ax[i, j].tick_params(axis='both', which='major', labelsize=12)

            # Make background grey if no data
            if not (ax[i, j].lines or ax[i, j].collections):
                ax[i, j].set_facecolor('gainsboro')
    
    # Plot labels
    for j, n in enumerate(ns):
        ax[-1, j].set_xlabel(r'$B$ $[\Phi_0/a^2]$', fontsize=12)
        if show_title: ax[0, j].set_title(f'n = {n}', fontsize=14)
    for i, U in enumerate(Us):
        ax[i, 0].set_ylabel(y_label, fontsize=12)
        if show_title: ax[i, -1].annotate(f'U = {U}', (1.02, 0.48), rotation=270, xycoords='axes fraction', fontsize=14)
    if show_legend: ax[0,-1].legend(labelspacing = 0.35, fontsize=9)
    
    plt.tight_layout()
    
######################## plots vs. T
def plot_rho_T(df, nfluxes, n, U, betas, Nx=None, Ny=None, tp=None, ylim=None, xlim=None, fmt='-', ax=None, show_legend=True, color_dict=None, labels=None, style={}):
    x_label = 'T/t'
    y_label = r'$\rho$ [$\hbar/e^2$]'
    Nx, Ny, tp, color_dict, ax = _setup_plot(df, nfluxes, x_label, y_label, Nx=Nx, Ny=Ny, tp=tp, color_dict=color_dict, ax=ax, palette='tab10')    
    
    for k, nflux in enumerate(nfluxes):
        df_subset = df[(df['Nx']==Nx) & (df['Ny']==Ny) & (df['tp']==tp) & (df['U']==U) & (df['nflux']==nflux) & (df['n']==n) & (df['beta'].isin(betas))].sort_values(by='beta', ascending=False).reset_index()
        if df_subset.empty:
            continue
        rhos = df_subset['rho_xx']
        rho_errs = df_subset['rho_xx_err']
        betas = df_subset['beta']
        Ts = 1/betas
        # ax.errorbar(Ts, rhos, yerr=rho_errs, label=labels[k] if labels is not None else rf'$B$={nflux}/{Nx*Ny}', fmt=fmt, markersize=2, color = color_dict[nflux], elinewidth=1, capsize=0.8, **style)
        ax.errorbar(Ts, rhos, yerr=rho_errs, label=labels[k] if labels is not None else rf'$B$={nflux}/{Nx*Ny}', color = color_dict[nflux], **style)
                
    # Plot labels
    if show_legend: ax.legend()
    plt.tight_layout()

def plot_dc_sig_T(df, nfluxes, ns, Us, betas, Nx=None, Ny=None, tp=None, mode='xx', ylim=None, xlim=None, ax=None, fmt='-', show_legend=True, colors=None, e2_h=False, labels=None):
    if mode=='xx':
        sig_label = 'sig_xx'
        y_label = rf'$\sigma_{{xx}}(0)$'
    elif mode=='xy':
        sig_label = 'sig_xy'
        y_label = rf'$\sigma_{{xy}}(0)$'
    else:
        pass
        
    Nx = _resolve_param(df, Nx, "Nx")
    Ny = _resolve_param(df, Ny, "Ny")
    tp = _resolve_param(df, tp, "tp")

    if colors is None: colors = sns.color_palette('tab10', len(nfluxes))
    #fig, ax = plt.subplots(nrows=len(Us), ncols = len(ns), figsize=(default_figsize[0]*len(ns)*0.7, default_figsize[1]*1*len(Us)), dpi=dpi)
    if len(Us)==1 and len(ns)==1:
        figsize = (default_figsize[0], default_figsize[1]*1.5)
    else:
        figsize = (default_figsize[0]*len(ns)*0.7, default_figsize[1]*1*len(Us))
    # figsize = (default_figsize[0]*len(ns)*0.75, default_figsize[1]*len(Us))
    if ax is None:
        fig, ax = plt.subplots(nrows=len(Us), ncols = len(ns), figsize=figsize, sharey=True)
    ax = np.reshape(ax, (len(Us), len(ns)))
    
    for i, U in enumerate(Us):
        for j, n in enumerate(ns):
            for k, nflux in enumerate(nfluxes):
                df_subset = df[(df['Nx']==Nx) & (df['Ny']==Ny) & (df['tp']==tp) & (df['U']==U) & (df['nflux']==nflux) & (df['n']==n) & (df['beta'].isin(betas))].sort_values(by='beta', ascending=False)
                if df_subset.empty:
                    continue
                sigs = df_subset[sig_label]
                sig_errs = df_subset[sig_label+'_err']
                if e2_h:
                    sigs = sigs*2*np.pi
                    sig_errs = sig_errs*2*np.pi
                betas_plot = df_subset['beta']
                Ts_plot = 1/betas_plot
                lw = 1 if ':' in fmt else 0.75
                ax[i, j].errorbar(Ts_plot, sigs, yerr=sig_errs, label=labels[k] if labels is not None else rf'$B$={nflux}/{Nx*Ny}', fmt=fmt, marker='o', markersize=2, color = colors[list(nfluxes).index(nflux)], lw=lw, elinewidth=1, capsize=0.8)
            
            # Remove x and y axis labels from interior axes
            if j!=0: ax[i, j].tick_params(labelleft=False)
            if i!=ax.shape[0]-1: ax[i, j].tick_params(labelbottom=False)

            # Make background grey if no data
            if not (ax[i, j].lines or ax[i, j].collections):
                ax[i, j].set_facecolor('gainsboro')

    for j, n in enumerate(ns):
        ax[-1, j].set_xlabel('T/t', fontsize=12)
        # ax[0, j].set_title(f'n = {n}', fontsize=14)
    for i, U in enumerate(Us):
        ax[i, 0].set_ylabel(y_label, fontsize=12)
        # ax[i, -1].annotate(f'U = {U}', (1.02, 0.48), rotation=270, xycoords='axes fraction', fontsize=14)
    if show_legend: ax[0,-1].legend(labelspacing = 0.35, fontsize=9)


def plot_MR_T(df, nfluxes, ns, Us, betas, Nx=None, Ny=None, tp=None, mode='xx', ylim=None, xlim=None, ax=None, fmt='-', labels=None, colors=None, show_legend=True):
    if mode=='xx':
        sig_label = 'sig_xx'
        y_label = rf'$\sigma_{{xx}}(0)$'
    elif mode=='xy':
        sig_label = 'sig_xy'
        y_label = rf'$\sigma_{{xy}}(0)$'
    else:
        pass
        
    Nx = _resolve_param(df, Nx, "Nx")
    Ny = _resolve_param(df, Ny, "Ny")
    tp = _resolve_param(df, tp, "tp")
    
    if colors is None: colors = sns.color_palette('tab10', len(nfluxes))
    #fig, ax = plt.subplots(nrows=len(Us), ncols = len(ns), figsize=(default_figsize[0]*len(ns)*0.7, default_figsize[1]*1*len(Us)), dpi=dpi)
    if len(Us)==1 and len(ns)==1:
        figsize = (default_figsize[0], default_figsize[1]*1.5)
    else:
        figsize = (default_figsize[0]*len(ns)*0.7, default_figsize[1]*1*len(Us))
    # figsize = (default_figsize[0]*len(ns)*0.75, default_figsize[1]*len(Us))
    if ax is None:
        fig, ax = plt.subplots(nrows=len(Us), ncols = len(ns), figsize=figsize, sharey=True)
    ax = np.reshape(ax, (len(Us), len(ns)))
    
    for i, U in enumerate(Us):
        for j, n in enumerate(ns):
            for k, nflux in enumerate(nfluxes):
                df_subset = df[(df['Nx']==Nx) & (df['Ny']==Ny) & (df['tp']==tp) & (df['U']==U) & (df['nflux']==nflux) & (df['n']==n) & (df['beta'].isin(betas))].sort_values(by='beta', ascending=False)
                if df_subset.empty:
                    continue
                MRs = df_subset['MR']*100
                MR_errs = df_subset['MR_err']*100
                betas_plot = df_subset['beta']
                Ts_plot = 1/betas_plot
                lw = 1 if ':' in fmt else 0.75
                ax[i, j].errorbar(Ts_plot, MRs, yerr=MR_errs, label=labels[k] if labels is not None else rf'$B$={nflux}/{Nx*Ny}', fmt=fmt, markersize=2, color = colors[list(nfluxes).index(nflux)], lw=lw, elinewidth=1, capsize=0.8)
            
            # Remove x and y axis labels from interior axes
            if j!=0: ax[i, j].tick_params(labelleft=False)
            if i!=ax.shape[0]-1: ax[i, j].tick_params(labelbottom=False)

            # Make background grey if no data
            if not (ax[i, j].lines or ax[i, j].collections):
                ax[i, j].set_facecolor('gainsboro')

    for j, n in enumerate(ns):
        ax[-1, j].set_xlabel('T/t', fontsize=12)
        # ax[0, j].set_title(f'n = {n}', fontsize=14)
    for i, U in enumerate(Us):
        ax[i, 0].set_ylabel('MR [%]', fontsize=12)
        # ax[i, -1].annotate(f'U = {U}', (1.02, 0.48), rotation=270, xycoords='axes fraction', fontsize=14)
    if show_legend: ax[0,-1].legend(labelspacing = 0.35, fontsize=9)

def plot_doubleocc_T(df, nfluxes, n, U, betas,  Nx=None, Ny=None, tp=None, ylim=None, xlim=None, ax=None, show_legend=True, color_dict=None, labels=None, style={}):
    x_label = 'T/t'
    y_label = r'$\langle n_{\uparrow} n_{\downarrow} \rangle$'
    y_label = r'$\langle n_{\uparrow}\! n_{\downarrow} \rangle$'
    Nx, Ny, tp, color_dict, ax = _setup_plot(df, nfluxes, x_label, y_label, Nx=Nx, Ny=Ny, tp=tp, color_dict=color_dict, ax=ax, palette='tab10')    

    for k, nflux in enumerate(nfluxes):
        df_subset = df[(df['Nx']==Nx) & (df['Ny']==Ny) & (df['tp']==tp) & (df['U']==U) & (df['nflux']==nflux) & (df['n']==n) & (df['beta'].isin(betas))].sort_values(by='beta', ascending=False)
        if df_subset.empty:
            continue
        double_occs = df_subset['double_occ']
        double_occ_errs = df_subset['double_occ_err']
        betas_plot = df_subset['beta']
        Ts_plot = 1/betas_plot
        ax.errorbar(Ts_plot, double_occs, yerr=double_occ_errs, label=labels[k] if labels is not None else rf'$B$={nflux}/{Nx*Ny}', color=color_dict[nflux], **style)
    
    if show_legend: ax.legend()

############## plot sigma spectra directly
def plot_sig_spectra(pickle_folder, nflux, n, U, betas, sigs_plot=['re_sig_xx', 'im_sig_xy', 're_sig_xy'], bs_mode='errorbar', B0=0, ylim=None, xlim=None, suptitle='', ax=None, colors=None):
    # temperature resolved, all for given nflux, n, U
    # suppose for now the only utility is to check out how spectra change over temperatures, maybe useful for particularly bad data points
    if ax is None:
        fig, ax = plt.subplots(ncols=len(sigs_plot), figsize=(default_figsize[0]*len(sigs_plot), default_figsize[1]*1))
    # df_subset = df[(df['U']==U) & (df['nflux']==nflux) & (df['n']==n)].sort_values(by='beta')
    
    if colors is None:
        cmap = plt.get_cmap('coolwarm')
        colors = [cmap(i) for i in np.linspace(0, 1, len(betas))][::-1]
        
    for i, beta in enumerate(betas):
        sig = oc.get_sig_pickle(pickle_folder, nflux=nflux, n=n, U=U, beta=beta)
        # print(np.mean(sig.sign))
        for j, sig_plot in enumerate(sigs_plot):
            oc.plot_sigma(sig, ax[j], sig_plot, color=colors[i], bs_mode=bs_mode, show_opt_info=False, label=rf'$\beta=${beta}')
    
    ax[-1].legend(loc='upper right')
    if xlim: [axes.set_xlim(xlim) for axes in ax]
    if ylim: [axes.set_ylim(ylim) for axes in ax]
    fig.suptitle(f"{pickle_folder.split('/')[-2]}, nflux={nflux}, n={n}, U={U}")

    plt.tight_layout()

############## boo boo kohler
# def plot_MR(df, nfluxes, ns, Us, betas, Nx=None, Ny=None, tp=None, mode='rho', shared_axes=True, ylim=None, xlim=None, suptitle='', calc_fit=False, loglog=False, fmt='-', ax=None, show_legend=True):
def plot_kohler2(df, nfluxes, n, U, betas, Nx=None, Ny=None, tp=None, calc_fit=False, ax=None, suptitle='', xlim=None, ylim=None, show_legend=True, color_dict=None, style={}):
    x_label = r'$B/\rho_0$'
    y_label = 'MR [%]'
    Nx, Ny, tp, color_dict, ax = _setup_plot(df, nfluxes, x_label, y_label, Nx=Nx, Ny=Ny, tp=tp, color_dict=color_dict, ax=ax, palette='tab10')    

    if calc_fit:
        print("Fitting MRs to power law A*x^B")
        print(f"{'n':>6}  {'U':>4}  {'beta':>4}  {'A':>10} ± {'dA':<10}  {'B':>10} ± {'dB':<10}")
        print("-" * 84)
    
    for k, beta in enumerate(betas):
        # Get subset of df for specific U, beta, n
        df_subset = df[(df['Nx']==Nx) & (df['Ny']==Ny) & (df['tp']==tp) & (df['U']==U) & (df['beta']==beta) & (df['n']==n)].sort_values(by='nflux').reset_index() # do not change this lol
        if df_subset.empty:
            continue

        rho_B0 = df_subset.query('nflux==0')['rho_xx'].item()
        rho_B0_err = df_subset.query('nflux==0')['rho_xx_err'].item()
        MRs = df_subset['MR']*100
        MR_errs = df_subset['MR_err']*100
        rho_B0_uc = ufloat(rho_B0, rho_B0_err)

        mask = df_subset['nflux'].isin(nfluxes)
        nfluxes_plot = np.array(df_subset['nflux'])[mask]
        Bs_plot = nfluxes_plot/(Nx*Ny)
        B_errs = np.array([((nflux/(Nx*Ny))/rho_B0_uc).std_dev for nflux in nfluxes_plot])
        MRs_plot = np.array(MRs)[mask]
        MR_errs_plot = np.array(MR_errs)[mask]
        # mask = np.array(MRs)>0

        ax.errorbar(Bs_plot/rho_B0, MRs_plot, yerr=MR_errs_plot, xerr=B_errs, label=rf'$\beta$={beta}', color=color_dict[beta], **style)
        # ax[i, j].set_yscale('log', nonpositive='clip')
        # ax[i, j].set_xscale('log', nonpositive='clip')

        if ylim is not None:
            ax.set_ylim(*ylim)
        if xlim is not None:
            ax.set_xlim(*xlim)

        if calc_fit:
            def fit_func(logx, logA, C):
                return logA + C*logx
            mask = nfluxes_plot > 0 # exclude (0,0) point
            params, cov = scipy.optimize.curve_fit(fit_func, np.log(nfluxes_plot[mask]/(Nx*Ny)/rho_B0), np.log(MRs[mask]), p0=[1, 2], sigma=MR_errs[mask]/MRs[mask], absolute_sigma=True)
            perr = np.sqrt(np.diag(cov))
            
            print(f"{n:>6}  {U:>4}  {beta:>4}  {np.exp(params[0]):10.2f} ± {np.exp(params[0])*perr[0]:<10.2f}  "
            f"{params[1]:10.3f} ± {perr[1]:<10.3f}")

    if show_legend: ax.legend()

def plot_sigma_xy(df, Us, ns, betas, Bs, ylims=None, xlim=None, suptitle=''):
    # wait this doesn't make sense bc there's so many spectra per temperature lol idk
    pass


######################## comparison plots
def compare_sigs(rho_dfs, nfluxes, ns, Us, betas, xlim=None, B0=0):
    ### just gonna have this produce a bunch of different plots for now
    figsize = (default_figsize[0]*1.1, default_figsize[1]*1.5)
    fmts = ['-', ':', '.', '--']
    
    ### compare sig_xx and sig_xy
    fig, ax = plt.subplots(nrows=len(Us), ncols = len(ns), figsize=figsize, sharey=True)
    for i, rho_df in enumerate(rho_dfs):
        plot_dc_sig_B(rho_df, nfluxes, ns, Us, betas, mode='xx', xlim=xlim, ax=ax, fmt=fmts[i], show_legend=not i)

    fig, ax = plt.subplots(nrows=len(Us), ncols = len(ns), figsize=figsize, sharey=True)
    for i, rho_df in enumerate(rho_dfs):
        plot_dc_sig_B(rho_df, nfluxes, ns, Us, betas, mode='xy', xlim=xlim, ax=ax, fmt=fmts[i], show_legend=not i)

    ### compare rho vs. B
    fig, ax = plt.subplots(nrows=len(Us), ncols = len(ns), figsize=figsize, sharey=True)
    for i, rho_df in enumerate(rho_dfs):
        plot_rho_B(rho_df, nfluxes, ns, Us, betas, mode='rho', xlim=xlim, ax=ax, fmt=fmts[i], show_legend=not i)

    ### compare MR vs. B
    fig, ax = plt.subplots(nrows=len(Us), ncols = len(ns), figsize=figsize, sharey=True)
    for i, rho_df in enumerate(rho_dfs):
        df = calc_MR_df(rho_df, B0=B0)
        plot_MR(df, nfluxes, ns, Us, betas, mode='rho', xlim=xlim, ax=ax, fmt=fmts[i], show_legend=not i)

    ### compare MR vs. T, only one nflux point
    fig, ax = plt.subplots(nrows=len(Us), ncols = len(ns), figsize=figsize, sharey=True)
    for i, rho_df in enumerate(rho_dfs):
        df = calc_MR_df(rho_df, B0=B0)
        plot_MR_T(df, [3], ns, Us, betas, mode='rho', xlim=xlim, ax=ax, fmt=fmts[i], show_legend=not i)
        
    # fig, ax = plt.subplots(nrows=len(Us), ncols = len(ns), figsize=figsize, sharey=True)
    # for i, rho_df in enumerate(rho_dfs):
    #     plot_rho_T(rho_df, nfluxes, ns, Us, ax=ax, fmt=fmts[i], show_legend=not i)

    

def compare_sig_spectra(pickle_folders, nfluxes, n, U, beta, sigs_plot=['re_sig_xx', 'im_sig_xy','re_sig_xy'], bs_mode='errorbar', B0=0, labels=None, ylim=None, xlim=None, suptitle=''):
    """Plots sig vs. w for same nflux*, n, U, beta, different pickle folders. Can compare different lattice sizes or t'. """
    ### *has to be nfluxes bc the closest B field is not always exact/same nflux
    if labels is None: labels = ['']*len(pickle_folders)    
    fig, ax = plt.subplots(ncols=len(sigs_plot), figsize=(default_figsize[0]*len(sigs_plot), default_figsize[1]*1))
    colors = sns.color_palette('husl', len(pickle_folders))
    for i, pickle_folder in enumerate(pickle_folders):
        sig = oc.get_sig_pickle(pickle_folder, nflux=nfluxes[i], n=n, U=U, beta=beta)
        for j, sig_plot in enumerate(sigs_plot):
            oc.plot_sigma(sig, ax[j], sig_plot, color=colors[i], bs_mode=bs_mode, show_opt_info=False, label=labels[i])
    
    ax[-1].legend(loc='upper right')
    if xlim: [axes.set_xlim(xlim) for axes in ax]
    if ylim: [axes.set_ylim(ylim) for axes in ax]
    # fig.suptitle(f"{pickle_folder.split('/')[-2]}, nflux={nflux}, n={n}, U={U}")
    plt.tight_layout()

