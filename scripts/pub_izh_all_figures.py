#!/usr/bin/env python3
"""
pub_izh_all_figures.py — Publication-quality figures for Izhikevich intermittency.

Generates five separate figures (PDF + PNG at 600 DPI):
  1. pub_izh_M.pdf        — M(u') function, 15 d-values, 5×3 grid
  2. pub_izh_phi.pdf       — φ(u') RPD, 15 d-values, 5×3 grid
  3. pub_izh_l_map.pdf     — l(u') laminar-length map, 15 d-values, 5×3 grid
  4. pub_izh_pdll.pdf      — ψ(l) PDLL, 15 d-values, 5×3 grid
  5. pub_izh_scaling.pdf   — ⟨l⟩ vs ε scaling, single panel

Style: ASME J. Comp. Nonlin. Dyn. (serif, inward ticks, no grid, no titles).
Both c_lam=1.65 and 2.0 overlaid in each panel.
No legends — distinction described in figure captions.
"""
import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(BASE, 'data')
FIG  = os.path.join(BASE, 'figures')
LOGS = os.path.join(BASE, 'data')

# ═══════════════════════════════════════════════════════════════════════════════
# rcParams — ASME journal style
# ═══════════════════════════════════════════════════════════════════════════════
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'mathtext.fontset': 'cm',
    'text.usetex': False,
    'font.size': 8,
    'axes.labelsize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'xtick.minor.size': 1.8,
    'ytick.minor.size': 1.8,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'axes.linewidth': 0.6,
    'axes.grid': False,
    'lines.linewidth': 1.0,
    'lines.markersize': 3,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'legend.frameon': False,
})

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
DVALS = ['-11.86', '-11.90', '-11.94', '-12.00', '-12.06',
         '-12.12', '-12.20', '-12.25', '-12.35', '-12.45',
         '-12.55', '-12.60', '-12.80', '-12.90', '-13.00']

CLAMS = [1.65, 2.0]
FIT_W = 0.5
NROWS, NCOLS = 5, 3

C_165 = '#0072B2'   # blue  (Okabe-Ito)
C_200 = '#D55E00'   # vermillion
CLAM_COLOR = {1.65: C_165, 2.0: C_200}
CLAM_MARKER = {1.65: 'o', 2.0: '^'}
CLAM_LS = {1.65: '-', 2.0: '--'}

# Branch breakpoints for piecewise RPD (from M_rpd_theory_vs_num.py)
_BK_165 = {
    '-11.86': [-0.687, 0.798], '-11.90': [-0.715, 0.825],
    '-11.94': [-0.715, 0.853], '-12.00': [-0.745, 0.908],
    '-12.06': [-0.797, 0.963], '-12.12': [-0.825, 1.018],
    '-12.20': [-0.852, 1.100], '-12.25': [-0.880, 1.127],
    '-12.35': [-0.935, 1.238], '-12.45': [-0.990, 1.190, 1.365],
    '-12.55': [-1.045, 1.495], '-12.60': [-1.072, 1.265, 1.562],
    '-12.80': [-1.180, -0.850, 0.830],
    '-12.90': [-1.237], '-13.00': [-1.441, -1.290],
}
_BK_200 = {
    '-11.86': [-0.667, 0.789], '-11.90': [-0.700, 0.820],
    '-11.94': [-0.733, 0.840], '-12.00': [-0.750, 0.900],
    '-12.06': [-0.750, 0.967], '-12.12': [-0.820, 1.010],
    '-12.20': [-0.867, 1.100], '-12.25': [-0.900, 1.120],
    '-12.35': [-0.950, 1.240], '-12.45': [-1.000, 1.200, 1.350],
    '-12.55': [-1.040, 1.500], '-12.60': [-1.067, 1.267, 1.560],
    '-12.80': [-1.180, -0.880, 0.850, 1.850],
    '-12.90': [-1.233], '-13.00': [-1.450, -1.300],
}
BKPTS = {}
for _d in DVALS:
    BKPTS[(_d, 1.65)] = _BK_165[_d]
    BKPTS[(_d, 2.0)]  = _BK_200[_d]

PAD = 0.05

# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_ep(d, clam):
    ep = np.loadtxt(os.path.join(DATA, f'd{d}_ep_clam{clam:.2f}.dat'), comments='#')
    return ep[:, 0], ep[:, 1], ep[:, 2].astype(int)

def load_meta(d):
    meta = np.loadtxt(os.path.join(DATA, f'd{d}_meta.txt'))
    return meta[0], meta[1]

def load_reinj_all(d, clam):
    """All reinjection points (both branches concat) for M/φ."""
    r0, r1, _ = load_ep(d, clam)
    u = np.concatenate([r0, r1])
    return u[(u > -clam) & (u < clam)]

def load_entry_branch(d, clam):
    """Entry-branch reinj + l_iter for l-map and PDLL."""
    r0, _, l_iter = load_ep(d, clam)
    mask = (np.abs(r0) < clam) & (np.abs(r0) > 1e-6)
    return r0[mask], l_iter[mask]


# ═══════════════════════════════════════════════════════════════════════════════
# Cubic drift fit for l(u') theory
# ═══════════════════════════════════════════════════════════════════════════════

def fit_cubic(d):
    dat = np.loadtxt(os.path.join(DATA, f'd{d}_mapcurve.dat'), comments='#')
    s, y = dat[:, 0], dat[:, 1]
    m = np.abs(s) < FIT_W
    if m.sum() < 10:
        m = np.abs(s) < 1.0
    M = np.column_stack([s[m], s[m]**3])
    (C, A), *_ = np.linalg.lstsq(M, y[m], rcond=None)
    return C, A

def l_cont(u, eps, A, c):
    u2 = np.clip(u**2, 1e-8, None)
    arg = c**2 * (eps + A * u2) / (u2 * (eps + A * c**2))
    return np.log(np.clip(arg, 1.0, None)) / (2.0 * eps)

def l_ceil(u, eps, A, c):
    return np.maximum(1, np.ceil(l_cont(u, eps, A, c)))


# ═══════════════════════════════════════════════════════════════════════════════
# Piecewise RPD theory for M(u') and φ(u')
# ═══════════════════════════════════════════════════════════════════════════════

def _fit_branch(pts, lo, hi, inverted=False, is_lo_bnd=False, is_hi_bnd=False):
    bp = pts[(pts > lo) & (pts < hi)] if inverted else pts[(pts >= lo) & (pts < hi)]
    if len(bp) < 20:
        return None
    bp_s = np.sort(bp)[::-1] if inverted else np.sort(bp)
    if inverted:
        refl = hi + (hi - bp_s)
        M_arr = np.cumsum(refl) / np.arange(1, len(refl)+1, dtype=float)
    else:
        M_arr = np.cumsum(bp_s) / np.arange(1, len(bp_s)+1, dtype=float)
    flo = lo if is_lo_bnd else lo + PAD
    fhi = hi if is_hi_bnd else hi - PAD
    if inverted:
        fm = (bp_s <= flo) & (bp_s >= fhi)
    else:
        fm = (bp_s >= flo) & (bp_s <= fhi)
    if fm.sum() < 20:
        fm = np.ones(len(bp_s), dtype=bool)
    s = stats.linregress(bp_s[fm], M_arr[fm])
    return dict(u=bp_s, M=M_arr, m=s.slope, n=len(bp), inverted=inverted)


def _alpha(m, inv):
    if inv:
        return (-2*m - 1) / (1 + m) if abs(1 + m) > 1e-10 else np.nan
    return (2*m - 1) / (1 - m) if abs(1 - m) > 1e-10 else np.nan


def build_theory_M_phi(d, clam, ngrid=4000):
    """Return (x, phi_theo, M_theo) arrays for theoretical RPD and M."""
    pts = load_reinj_all(d, clam)
    bk = BKPTS[(d, clam)]
    edges = [-clam] + list(bk) + [clam]
    branches = []
    for i in range(len(edges)-1):
        inv = (i == 0)
        br = _fit_branch(pts, edges[i], edges[i+1], inverted=inv,
                         is_lo_bnd=(i==0), is_hi_bnd=(i==len(edges)-2))
        if br and not inv and br['m'] > 0.5:
            br = _fit_branch(pts, edges[i], edges[i+1], inverted=True,
                             is_lo_bnd=(i==0), is_hi_bnd=(i==len(edges)-2))
        branches.append(br)

    N_total = sum(br['n'] for br in branches if br is not None)
    x = np.linspace(-clam, clam, ngrid)
    dx = x[1] - x[0]
    phi = np.zeros(ngrid)
    for j, br in enumerate(branches):
        if br is None:
            continue
        lo, hi = edges[j], edges[j+1]
        alpha = _alpha(br['m'], br['inverted'])
        f_j = br['n'] / N_total
        mask = (x >= lo) & (x <= hi)
        xm = x[mask]
        dist = (hi - xm) if br['inverted'] else (xm - lo)
        dist = np.maximum(dist, 1e-15)
        I_j = (hi - lo)**(alpha + 1) / (alpha + 1)
        phi[mask] = (f_j / I_j) * dist**alpha

    P = np.cumsum(phi * dx)
    Q = np.cumsum(x * phi * dx)
    M_theo = np.where(P > 1e-15, Q / P, x)
    return x, phi, M_theo


# ═══════════════════════════════════════════════════════════════════════════════
# PDLL ψ(l) computation
# ═══════════════════════════════════════════════════════════════════════════════

def pmf(lvals, lmax):
    li = np.clip(np.rint(lvals).astype(int), 1, lmax)
    counts = np.bincount(li, minlength=lmax + 1)[1:lmax + 1]
    s = counts.sum()
    return np.arange(1, lmax + 1), (counts / s if s > 0 else counts.astype(float))


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: annotate d-value in panel
# ═══════════════════════════════════════════════════════════════════════════════

def _label_d(ax, d):
    ax.text(0.05, 0.92, f'$d={d}$', transform=ax.transAxes,
            fontsize=6, va='top', ha='left')


def _edge_labels(axes, xlabel, ylabel, nrows, ncols):
    """Set axis labels only on edge panels."""
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            if i == nrows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)


def save_fig(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ['pdf', 'png']:
        p = os.path.join(FIG, f'{name}.{ext}')
        fig.savefig(p)
        print(f'  {p}')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: M(u') — 5×3 grid
# ═══════════════════════════════════════════════════════════════════════════════

def fig_M():
    print('\n── Figure: M(u\') ──')
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(7.0, 8.5),
                              sharex=False, sharey=False)
    for idx, d in enumerate(DVALS):
        r, c = divmod(idx, NCOLS)
        ax = axes[r, c]
        for clam in CLAMS:
            col = CLAM_COLOR[clam]
            pts = load_reinj_all(d, clam)
            u_s = np.sort(pts)
            M_s = np.cumsum(u_s) / np.arange(1, len(u_s)+1, dtype=float)
            step = max(1, len(u_s) // 1500)
            ax.plot(u_s[::step], M_s[::step], '.', color=col, ms=0.4,
                    alpha=0.35, rasterized=True)
            x_th, _, M_th = build_theory_M_phi(d, clam, ngrid=3000)
            ax.plot(x_th, M_th, CLAM_LS[clam], color=col, lw=0.8, alpha=0.9)
        _label_d(ax, d)
    _edge_labels(axes, r"$u'$", r'$M(u\')$', NROWS, NCOLS)
    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    save_fig(fig, 'pub_izh_M')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: φ(u') RPD — 5×3 grid
# ═══════════════════════════════════════════════════════════════════════════════

def fig_phi():
    print('\n── Figure: φ(u\') ──')
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(7.0, 8.5),
                              sharex=False, sharey=False)
    for idx, d in enumerate(DVALS):
        r, c = divmod(idx, NCOLS)
        ax = axes[r, c]
        for clam in CLAMS:
            col = CLAM_COLOR[clam]
            pts = load_reinj_all(d, clam)
            nbins = 120
            counts, edges = np.histogram(pts, bins=nbins, range=(-clam, clam))
            centers = 0.5 * (edges[:-1] + edges[1:])
            density = counts / (counts.sum() * (edges[1] - edges[0]))
            ax.step(centers, density, where='mid', color=col, lw=0.6,
                    alpha=0.7)
            x_th, phi_th, _ = build_theory_M_phi(d, clam, ngrid=3000)
            phi_clip = np.clip(phi_th, 0,
                               density.max() * 2.5 if density.max() > 0 else 5)
            ax.plot(x_th, phi_clip, CLAM_LS[clam], color=col, lw=0.8,
                    alpha=0.9)
        ax.set_xlim(-2.1, 2.1)
        ax.set_ylim(bottom=0)
        _label_d(ax, d)
    _edge_labels(axes, r"$u'$", r"$\varphi(u')$", NROWS, NCOLS)
    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    save_fig(fig, 'pub_izh_phi')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: l(u') laminar-length map — 5×3 grid
# ═══════════════════════════════════════════════════════════════════════════════

def fig_l_map():
    print('\n── Figure: l(u\') ──')
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(7.0, 8.5),
                              sharex=False, sharey=False)
    for idx, d in enumerate(DVALS):
        r, c = divmod(idx, NCOLS)
        ax = axes[r, c]
        C_fit, A_fit = fit_cubic(d)
        eps = C_fit - 1.0

        for clam in CLAMS:
            col = CLAM_COLOR[clam]
            u_reinj, l_sim = load_entry_branch(d, clam)
            step = max(1, len(u_reinj) // 1200)
            ax.scatter(u_reinj[::step], l_sim[::step], s=0.3, color=col,
                       alpha=0.25, rasterized=True, edgecolors='none')
            ug = np.linspace(-clam, clam, 500)
            ug = ug[np.abs(ug) > 0.02]
            lth = l_cont(ug, eps, A_fit, clam)
            ax.plot(ug, lth, CLAM_LS[clam], color=col, lw=0.7, alpha=0.9)

        lmax_plot = int(np.percentile(l_sim, 99)) + 1
        ax.set_yscale('log')
        ax.set_ylim(0.8, max(lmax_plot * 1.5, 10))
        ax.set_xlim(-2.1, 2.1)
        _label_d(ax, d)
    _edge_labels(axes, r"$u'$", r'$l$', NROWS, NCOLS)
    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    save_fig(fig, 'pub_izh_l_map')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: ψ(l) PDLL — 5×3 grid
# ═══════════════════════════════════════════════════════════════════════════════

def fig_pdll():
    print('\n── Figure: ψ(l) ──')
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(7.0, 8.5),
                              sharex=False, sharey=False)
    for idx, d in enumerate(DVALS):
        r, c = divmod(idx, NCOLS)
        ax = axes[r, c]
        C_fit, A_fit = fit_cubic(d)
        eps = C_fit - 1.0

        for clam in CLAMS:
            col = CLAM_COLOR[clam]
            mk = CLAM_MARKER[clam]
            ls = CLAM_LS[clam]
            u_reinj, l_sim = load_entry_branch(d, clam)
            lmax_h = max(int(l_sim.max()), 5)
            l_arr, psi_num = pmf(l_sim, lmax_h)

            lc = l_ceil(u_reinj, eps, A_fit, clam)
            _, psi_th = pmf(lc, lmax_h)

            lmax_plot = int(np.percentile(l_sim, 99.5)) + 1
            clip = l_arr <= lmax_plot
            pos_n = psi_num > 0
            pos_t = psi_th > 0

            ax.plot(l_arr[clip & pos_n], psi_num[clip & pos_n], mk,
                    color=col, ms=2.0, alpha=0.7, markeredgewidth=0.0)
            ax.plot(l_arr[clip & pos_t], psi_th[clip & pos_t], ls,
                    color=col, lw=0.7, alpha=0.8)

        ax.set_yscale('log')
        pmin = psi_num[psi_num > 0].min() if (psi_num > 0).any() else 1e-5
        ax.set_ylim(max(pmin * 0.3, 1e-6), 1.0)
        _label_d(ax, d)
    _edge_labels(axes, r'$l$', r'$\psi(l)$', NROWS, NCOLS)
    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    save_fig(fig, 'pub_izh_pdll')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: Scaling ⟨l⟩ vs ε — single panel
# ═══════════════════════════════════════════════════════════════════════════════

def fig_scaling():
    print('\n── Figure: scaling ──')
    fig, ax = plt.subplots(figsize=(3.54, 2.8))

    for clam in CLAMS:
        col = CLAM_COLOR[clam]
        mk = CLAM_MARKER[clam]
        ls = CLAM_LS[clam]
        eps_list, sim_list, th_list = [], [], []

        for d in DVALS:
            C_fit, A_fit = fit_cubic(d)
            eps = C_fit - 1.0
            u_reinj, l_sim = load_entry_branch(d, clam)
            mean_sim = l_sim.mean()
            mean_th = l_ceil(u_reinj, eps, A_fit, clam).mean()
            eps_list.append(eps)
            sim_list.append(mean_sim)
            th_list.append(mean_th)

        ea = np.array(eps_list)
        sa = np.array(sim_list)
        ta = np.array(th_list)

        ax.plot(ea, sa, mk, color=col, ms=4, markeredgecolor='k',
                markeredgewidth=0.3, zorder=5)
        ax.plot(ea, ta, ls, color=col, lw=1.2, zorder=3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\varepsilon_{\mathrm{local}} = C(d) - 1$')
    ax.set_ylabel(r'$\langle l \rangle$')
    fig.tight_layout(pad=0.3)
    save_fig(fig, 'pub_izh_scaling_v3')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(FIG, exist_ok=True)
    fig_M()
    fig_phi()
    fig_l_map()
    fig_pdll()
    fig_scaling()
    print('\nDone — all publication figures saved.')


if __name__ == '__main__':
    main()
