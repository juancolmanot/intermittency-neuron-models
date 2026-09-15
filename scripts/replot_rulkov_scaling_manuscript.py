#!/usr/bin/env python3
import numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'mathtext.fontset': 'cm',
    'font.size': 11,
    'axes.labelsize': 39,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'legend.fontsize': 22,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'legend.frameon': False,
    'legend.fontsize': 14,
})

COLOR_NUM = '#0000CC'
COLOR_TH  = '#CC0000'
A2 = 0.57446320

def phi_single(x, rpd):
    if x < rpd['x_hat'] or x > rpd['c']:
        return 0.0
    return rpd['b'] * (x - rpd['x_hat'])**rpd['alpha']

def l_type1(x_entry, c, a2, eps):
    if a2 * eps <= 0:
        return 0.0
    sqae = np.sqrt(a2 * eps)
    r = np.sqrt(a2 / eps)
    return (np.arctan(c * r) - np.arctan(x_entry * r)) / sqae

def mean_l_th(c, a2, eps, rpd):
    def integrand(x):
        return phi_single(x, rpd) * max(l_type1(x, c, a2, eps), 0.0)
    result, _ = quad(integrand, rpd['x_hat'], c, limit=500, epsabs=1e-12, epsrel=1e-10)
    return result

rpd_params = {
    0.1:  {'alpha': -0.520425, 'x_hat': -0.097792, 'b': 1.043223, 'c': 0.1},
    0.01: {'alpha': -0.955371, 'x_hat': -0.010000, 'b': 0.053142, 'c': 0.01},
}

for c_val in [0.1, 0.01]:
    c_str = f"{c_val:.2f}"
    data = np.loadtxt(f"{DATADIR}/rulkov_scaling_v2_c={c_str}.csv", delimiter=',', comments='#')
    eps = data[:, 1]
    mean_l = data[:, 3]
    n_ep = data[:, 6].astype(int)

    mask = (mean_l > 0) & (n_ep >= 100)
    eps_m, ml_m = eps[mask], mean_l[mask]

    rpd = rpd_params[c_val]
    ml_th_arr = np.array([mean_l_th(c_val, A2, e, rpd) for e in eps_m])

    log_e = np.log(eps_m)
    log_ml = np.log(ml_m)
    log_ml_th = np.log(ml_th_arr)

    c_num = np.polyfit(log_e, log_ml, 1)
    beta_num = -c_num[0]

    mask_th = ml_th_arr > 0
    c_th = np.polyfit(np.log(eps_m[mask_th]), np.log(ml_th_arr[mask_th]), 1)
    beta_th = -c_th[0]

    fit_x = np.linspace(log_e.min(), log_e.max(), 100)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(log_e, log_ml, 'o', color=COLOR_NUM, ms=7,
            label=f"Numerical ($\\beta={beta_num:.3f}$)")

    if np.any(mask_th):
        ax.plot(log_e[mask_th], log_ml_th[mask_th], '-', color=COLOR_TH, lw=3.0,
                label=f"Theoretical ($\\beta={beta_th:.3f}$)")

    ax.plot(fit_x, np.polyval(c_num, fit_x), '--', color=COLOR_NUM, lw=0.8, alpha=0.4)

    mid = np.median(log_e)
    mid_v = np.polyval(c_num, mid)
    ref_y = -0.5 * (fit_x - mid) + mid_v
    ax.plot(fit_x, ref_y, ':', color='k', lw=0.8, alpha=0.4,
            label=r'$\beta=0.5$')

    ax.set_xlabel(r'$\ln(\varepsilon)$')
    ax.set_ylabel(r'$\ln(\langle l \rangle)$')

    ax.legend(loc='upper right', frameon=False)
    ax.locator_params(axis='x', nbins=5); ax.locator_params(axis='y', nbins=5)


    plt.tight_layout()
    out = os.path.join(OUTDIR, f'rulkov_scaling_v2_c={c_str}.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")
