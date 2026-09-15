#!/usr/bin/env python3
"""
report_laminar_saturation.py

Didactic report: Why does ⟨l⟩ saturate as ε → 0 in this system?

Produces: reports/laminar_saturation_report.pdf  (multi-page figure set)
         reports/laminar_saturation_report.png   (composite)
"""
import os
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib import patheffects

BASEDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
FIGDIR  = os.path.join(BASEDIR, 'figures')
REPDIR  = os.path.join(BASEDIR, 'figures')
LOGDIR  = os.path.join(BASEDIR, 'data')
DATADIR = os.path.join(BASEDIR, 'data')
os.makedirs(REPDIR, exist_ok=True)

PUB_RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'mathtext.fontset': 'cm',
    'font.size': 16,
    'axes.labelsize': 28,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.4,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'legend.frameon': False,
    'legend.fontsize': 12,
    'text.usetex': True,
}
matplotlib.rcParams.update(PUB_RC)

DC   = -11.8145
CLAM = 2.0
A0   = 0.01482  # A from cubic fit at d=-11.90 (same d used in M/phi/psi figures)

# ---------------------------------------------------------------------------
# laminar length formula
# ---------------------------------------------------------------------------
def l_cubic(u0, eps, A, c):
    """l(u0; eps, A, c) = (1/2eps) ln[ c²(eps+Au0²) / (u0²(eps+Ac²) ) ]"""
    if eps <= 0 or abs(u0) < 1e-14 or abs(u0) >= c:
        return 0.0
    num = c**2 * (eps + A * u0**2)
    den = u0**2 * (eps + A * c**2)
    if num <= 0 or den <= 0:
        return 0.0
    return max(np.log(num / den) / (2.0 * eps), 0.0)

def l_inf(u0, A, c):
    """ε→0 saturation limit: l_∞ = (1/2A)(1/u0² − 1/c²)"""
    if abs(u0) >= c or abs(u0) < 1e-14:
        return 0.0
    return max((1.0 / (2 * A)) * (1.0 / u0**2 - 1.0 / c**2), 0.0)

def l_typeI(u0, eps, c):
    """Type-I (saddle-node) approximate laminar length: π/(2√ε) for u0≈0."""
    if eps <= 0:
        return np.inf
    return np.pi / (2.0 * np.sqrt(eps))

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_step3():
    path = os.path.join(LOGDIR, 'step3_theoretical_results.csv')
    data = np.genfromtxt(path, delimiter=',', names=True)
    return data

def load_scaling():
    path = os.path.join(DATADIR, 'izhikevich_scaling_v3.csv')
    data = np.genfromtxt(path, delimiter=',', comments='#')
    mask = data[:, 2] > 0
    return data[mask, 0], data[mask, 1], data[mask, 2]  # d, eps, mean_l

# ---------------------------------------------------------------------------
# Figure 1: Type-I vs Type-III — conceptual comparison
# ---------------------------------------------------------------------------
def fig1_conceptual():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    eps_vals = np.logspace(-4, -0.5, 300)

    # Type-I: ⟨l⟩ ~ π/(2√ε)
    l_I = np.pi / (2 * np.sqrt(eps_vals))

    # Type-III (this system): ⟨l⟩_th from Branch 1 integration (d=-11.90)
    # Branch 1 (inverted): φ₁ = b·(u_{c2} - u)^α₁ on [-c, u_{c2}]
    # Canonical Branch-1 parameters at d = -11.90: breakpoint and reflected-slope fit over
    # [-2, -0.73] from M_rpd_pdll_krause.get_branches/krause_theory (m_r = 0.389, alpha_1 = -0.364).
    # These are the values behind the theoretical curves of the M/RPD figures and give
    # <l>_inf(Branch 1) = 24.5, the value quoted in the manuscript. (An earlier full-branch fit,
    # alpha_1 = -0.422 with u_c2 = -0.65, is superseded.)
    u_c2 = -0.676
    alpha1 = -0.364
    b_norm_1 = 0.4701
    k1 = 1.0

    def mean_l_th(eps):
        L1 = u_c2 + CLAM
        I1n = L1**(alpha1+1) / (alpha1+1)
        b = 1.0 / I1n  # approximate normalization (Branch 1 dominates)
        def f1(u):
            t = u_c2 - u
            if t <= 0: return 0.
            return b * t**alpha1 * l_cubic(u, eps, A0, CLAM)
        kw = dict(limit=200, epsabs=1e-8, epsrel=1e-6)
        I1, _ = quad(f1, -CLAM+1e-10, u_c2-1e-10, **kw)
        return I1

    l_III = np.array([mean_l_th(e) for e in eps_vals])

    # Also compute ε→0 saturation
    def l_inf_integrated():
        L1 = u_c2 + CLAM
        I1n = L1**(alpha1+1)/(alpha1+1)
        b = 1.0 / I1n
        def f1(u):
            t = u_c2 - u
            if t <= 0: return 0.
            return b * t**alpha1 * l_inf(u, A0, CLAM)
        kw = dict(limit=200, epsabs=1e-8, epsrel=1e-6)
        I1, _ = quad(f1, -CLAM+1e-10, u_c2-1e-10, **kw)
        return I1

    l_sat = l_inf_integrated()

    # Panel (a): log-log of ⟨l⟩ vs ε
    ax = axes[0]
    ax.loglog(eps_vals, l_I, 'C1-', lw=2, label='Type-I: $\\langle l\\rangle \\sim \\varepsilon^{-1/2}$')
    ax.loglog(eps_vals, l_III, 'C0-', lw=2, label='Type-III (this system)')
    ax.axhline(l_sat, color='C0', ls='--', lw=1.2, alpha=0.7,
               label=f'Saturation $\\langle l\\rangle_\\infty \\approx {l_sat:.0f}$')
    ax.set_xlabel(r'$\varepsilon = d - d_c$')
    ax.set_ylabel(r'$\langle l \rangle$')
    ax.set_title('(a) Type-I vs Type-III: $\\langle l \\rangle$ scaling')
    ax.legend(loc='upper right')
    ax.set_xlim(eps_vals.min(), eps_vals.max())
    ax.text(0.04, 0.35, 'Type-I diverges\n$\\langle l \\rangle \\to \\infty$',
            transform=ax.transAxes, color='C1', fontsize=9, va='center')
    ax.text(0.04, 0.15, 'Type-III saturates\n$\\langle l \\rangle \\to$ const',
            transform=ax.transAxes, color='C0', fontsize=9, va='center')

    # Panel (b): local slope d(log⟨l⟩)/d(log ε)
    ax = axes[1]
    log_e = np.log(eps_vals)

    # local slope of type-I: should be -0.5
    slope_I = np.gradient(np.log(l_I), log_e)
    # local slope of type-III
    valid = l_III > 0
    slope_III = np.full_like(l_III, np.nan)
    if np.sum(valid) > 5:
        slope_III[valid] = np.gradient(np.log(l_III[valid]), log_e[valid])

    ax.semilogx(eps_vals, slope_I, 'C1-', lw=2, label='Type-I slope $= -1/2$')
    ax.semilogx(eps_vals[valid], slope_III[valid], 'C0-', lw=2, label='Type-III local slope')
    ax.axhline(-0.5, color='gray', ls=':', lw=1, alpha=0.6, label='$\\gamma=1/2$ reference')
    ax.axhline(0.0, color='k', ls='-', lw=0.5, alpha=0.3)
    ax.set_xlabel(r'$\varepsilon = d - d_c$')
    ax.set_ylabel(r'Local exponent $\gamma_{\rm loc} = -d\ln\langle l\rangle/d\ln\varepsilon$')
    ax.set_title('(b) Local scaling exponent vs $\\varepsilon$')
    ax.legend(loc='lower left')
    ax.set_ylim(-1.0, 0.1)
    ax.text(0.6, 0.18, 'Type-III: $\\gamma \\to 0$\nas $\\varepsilon \\to 0$',
            transform=ax.transAxes, color='C0', fontsize=9)

    fig.suptitle('Figure 1: Why $\\langle l \\rangle$ saturates — Type-I vs Type-III intermittency',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Figure 2: l(u₀; ε) surface — the cubic drift formula
# ---------------------------------------------------------------------------
def fig2_laminar_surface():
    matplotlib.rcParams.update(PUB_RC)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    u0_vals = np.linspace(-1.95, 1.95, 600)
    eps_show = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    lstyles_a = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1))]

    # Panel (a): annotate innermost (eps=1e-4) and outermost (eps=5e-2)
    ax = axes[0]
    ann_fs = 19
    ann_kw = dict(fontsize=ann_fs, fontweight='bold')
    for eps, ls in zip(eps_show, lstyles_a):
        lvals = np.array([l_cubic(u, eps, A0, CLAM) for u in u0_vals])
        lvals[lvals > 1000] = np.nan
        ax.plot(u0_vals, lvals, color='k', ls=ls, lw=1.5)
    l_inf_vals = np.array([l_inf(u, A0, CLAM) for u in u0_vals])
    ax.plot(u0_vals, l_inf_vals, 'r-', lw=2.0)
    ax.set_xlabel(r'$\bar{u}_0$')
    ax.set_ylabel(r'$l(\bar{u}_0;\,\varepsilon)$')
    ax.set_ylim(0, 150)
    ax.annotate(r'$\varepsilon\!=\!10^{-4}$', xy=(0.55, 115), color='red', **ann_kw)
    ax.annotate(r'$\varepsilon\!=\!5\!\times\!10^{-2}$', xy=(-0.35, 12), **ann_kw)
    ax.text(0.03, 0.95, r'(a)', transform=ax.transAxes, fontsize=18, va='top', fontweight='bold')

    # Panel (b): drop u0=1.0 (overlaps -1.0), change 0.6 → -0.6
    ax = axes[1]
    eps_range = np.logspace(-6, -1, 200)
    u0_cases_b = [-1.5, -1.0, -0.7, -0.6]
    lstyles_b = ['-', '--', '-.', ':']
    b_label_y = {-1.5: 1.5, -1.0: 16.0, -0.7: 46.0, -0.6: 67.0}
    b_label_va = {-1.5: 'bottom', -1.0: 'bottom', -0.7: 'bottom', -0.6: 'top'}
    for u0, ls in zip(u0_cases_b, lstyles_b):
        lv = np.array([l_cubic(u0, e, A0, CLAM) for e in eps_range])
        lv_inf = l_inf(u0, A0, CLAM)
        ax.semilogx(eps_range, lv, color='k', ls=ls, lw=1.5)
        ax.axhline(lv_inf, color='k', ls=ls, lw=0.6, alpha=0.4)
        ax.text(1.5e-6, b_label_y[u0], f'$\\bar{{u}}_0\\!=\\!{u0}$',
                va=b_label_va[u0], **ann_kw)
    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel(r'$l(\bar{u}_0;\,\varepsilon)$')
    ax.text(0.03, 0.95, r'(b)', transform=ax.transAxes, fontsize=18, va='top', fontweight='bold')

    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Figure 3: The limit l_∞ = (1/2A)(1/u₀² − 1/c²) and why it is finite
# ---------------------------------------------------------------------------
def _load_phi_from_data(d_val='-11.86'):
    """Load actual φ(ū) from the Krause theory pipeline for a representative d."""
    import sys as _sys
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    from M_rpd_pdll_krause import (
        load_pts, get_branches, krause_theory, BKPTS, INV_OVERRIDES,
    )
    eps_raw, pts = load_pts(d_val, CLAM)
    bkpts = BKPTS[d_val]
    inv_ov = INV_OVERRIDES.get(d_val)
    edges, branches = get_branches(pts, CLAM, bkpts, inv_override=inv_ov)
    x_theo, phi_theo, _, _, _, _ = krause_theory(branches, edges, CLAM, ngrid=4000)

    # Numerical histogram for comparison
    nbins = 200
    counts, rpd_edges = np.histogram(pts, bins=nbins, range=(-CLAM, CLAM))
    rpd_c = 0.5 * (rpd_edges[:-1] + rpd_edges[1:])
    dw = rpd_edges[1] - rpd_edges[0]
    rpd_d = counts / (counts.sum() * dw)

    return x_theo, phi_theo, rpd_c, rpd_d, edges


def fig3_limit_formula():
    matplotlib.rcParams.update(PUB_RC)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    u0_pos = np.linspace(0.05, 1.95, 500)

    # Panel (a): l_∞(u₀)
    ax = axes[0]
    lv = np.array([l_inf(u, A0, CLAM) for u in u0_pos])
    ax.plot(u0_pos, lv, 'k-', lw=2.0)

    ax.set_xlabel(r'$|\bar{u}_0|$')
    ax.set_ylabel(r'$l_\infty(\bar{u}_0)$')
    ax.set_ylim(0, 200)
    ax.text(0.03, 0.95, r'(c)', transform=ax.transAxes, fontsize=18, va='top', fontweight='bold')

    # Panel (b): actual φ from Krause pipeline
    ax = axes[1]
    x_theo, phi_theo, rpd_c, rpd_d, edges = _load_phi_from_data('-11.90')

    # Numerical histogram
    ax.plot(rpd_c, rpd_d, '.', color='blue', ms=4.5, alpha=0.6, rasterized=True)

    # Theoretical curve (clip divergence for display, break at breakpoint)
    ymax_display = 3.0
    phi_clip = np.clip(phi_theo, 0, ymax_display * 1.5)
    bk = edges[1]
    mask_b1 = (phi_clip > 0) & (x_theo < bk - 0.01)
    mask_b2 = (phi_clip > 0) & (x_theo > bk + 0.01)
    ax.plot(x_theo[mask_b1], phi_clip[mask_b1], 'r-', lw=1.5)
    ax.plot(x_theo[mask_b2], phi_clip[mask_b2], 'r-', lw=1.5)

    # Shade gap region
    bk = edges[1]  # single breakpoint
    ax.axvspan(bk, CLAM, alpha=0.08, color='red')

    ax.set_xlabel(r'$\bar{u}$')
    ax.set_ylabel(r'$\phi(\bar{u})$')
    ax.set_xlim(-CLAM, CLAM)
    ax.set_ylim(-0.05, ymax_display)
    ax.text(0.03, 0.95, r'(d)', transform=ax.transAxes, fontsize=18, va='top', fontweight='bold')

    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Figure 4: Simulation data and theory — the saturation in practice
# ---------------------------------------------------------------------------
def fig4_data_comparison():
    s3 = load_step3()
    d_sim, eps_sim, ml_sim = load_scaling()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel (a): ⟨l⟩ vs log(ε) — linear y axis to show saturation plateau
    ax = axes[0]
    mask_th = s3['mean_l_th'] > 0

    ax.semilogx(s3['eps_csv'][mask_th], s3['mean_l_sim'][mask_th], 'o',
                color='#0055AA', ms=5, label='Simulation $\\langle l \\rangle$', zorder=5)
    ax.semilogx(s3['eps_csv'][mask_th], s3['mean_l_th'][mask_th], 's-',
                color='#CC2200', ms=4, lw=1.2, label='Theory $\\langle l \\rangle_{\\rm th}$')

    # show saturation regime
    sat_eps = s3['eps_csv'][mask_th][:6]
    sat_sim = s3['mean_l_sim'][mask_th][:6]
    mean_sat = np.mean(sat_sim)
    ax.axhspan(mean_sat - 5, mean_sat + 8, alpha=0.08, color='C0')
    ax.text(1.5e-3, mean_sat + 9, 'Saturation plateau\n(no further growth)', fontsize=8.5,
            color='#0055AA')

    ax.set_xlabel(r'$\varepsilon = d - d_c$')
    ax.set_ylabel(r'$\langle l \rangle$')
    ax.set_title('(a) $\\langle l \\rangle$ vs $\\varepsilon$ — saturation plateau')
    ax.legend(loc='lower right')

    # Panel (b): log-log with local slope
    ax2 = axes[1]
    le = np.log10(s3['eps_csv'][mask_th])
    ll_sim = np.log10(s3['mean_l_sim'][mask_th])
    ll_th  = np.log10(s3['mean_l_th'][mask_th])

    ax2.plot(le, ll_sim, 'o', color='#0055AA', ms=5, label='Simulation')
    ax2.plot(le, ll_th, 's-', color='#CC2200', ms=4, lw=1.2, label='Theory')

    # Fit slope in mid-range (skip first ~8 and last ~4 points)
    mid = slice(8, -4)
    from scipy import stats as spstat
    sl, ic, rv, pv, se = spstat.linregress(le[mid], ll_sim[mid])
    x_fit = np.linspace(le[8], le[-5], 50)
    ax2.plot(x_fit, sl*x_fit + ic, '--', color='#0055AA', lw=1.2, alpha=0.7,
             label=f'Fit slope $= {sl:.3f}$ ($\\gamma \\approx {-sl:.2f}$)')

    # Guide for γ=1/2
    mid_x = np.median(le)
    mid_y = np.median(ll_sim)
    ax2.plot(x_fit, -0.5*(x_fit - mid_x) + mid_y, ':', color='gray', lw=1.2,
             label='$\\gamma=1/2$ guide')

    ax2.set_xlabel(r'$\log_{10}\,\varepsilon$')
    ax2.set_ylabel(r'$\log_{10}\,\langle l \rangle$')
    ax2.set_title('(b) Log-log plot — effective exponent')
    ax2.legend(fontsize=8.5)

    # Annotate saturation regime
    ax2.axvspan(le[0], le[6], alpha=0.08, color='C0')
    ax2.text(le[1], ll_sim.max()*0.97, 'Saturation\n$\\gamma\\to 0$',
             fontsize=8, ha='center', color='#0055AA')

    ax2.axvspan(le[6], le[-8], alpha=0.08, color='orange')
    ax2.text(le[12], (ll_sim.max()+ll_sim.min())/2, 'Crossover\n$\\gamma\\approx0.24$',
             fontsize=8, ha='center', color='darkorange')

    fig.suptitle('Figure 4: Simulation data — saturation and crossover regimes',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Figure 5: Physical narrative — two intermittency types sketched
# ---------------------------------------------------------------------------
def fig5_physical_sketch():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel (a): Type-I map near tangency — narrow channel → l ~ 1/√ε
    ax = axes[0]
    x = np.linspace(-0.5, 2.0, 400)

    eps_I_vals = [0.0, 0.02, 0.06, 0.15]
    cols_I = ['black', '#1166CC', '#3388EE', '#88BBFF']

    for eps_I, col, lbl in zip(eps_I_vals, cols_I,
                                ['$\\varepsilon=0$ (tangency)', '0.02', '0.06', '0.15']):
        f = x + eps_I + x**2
        mask = (f >= -0.1) & (f <= 2.1)
        ax.plot(x[mask], f[mask], color=col, lw=1.5, label=lbl)

    ax.plot([-0.5, 2.0], [-0.5, 2.0], 'k-', lw=0.8, alpha=0.5)  # diagonal
    ax.axvspan(-0.5, 0.05, alpha=0.07, color='gray')
    ax.text(-0.15, 1.6, 'Narrow\nchannel', fontsize=8.5, ha='center', color='gray')
    ax.set_xlim(-0.5, 2.0)
    ax.set_ylim(-0.1, 2.1)
    ax.set_xlabel('$x_n$')
    ax.set_ylabel('$f(x_n)$ or $x_{n+1}$')
    ax.set_title('(a) Type-I (saddle-node): tangency at $\\varepsilon=0$\n'
                 r'Channel $\to 0$; $\langle l \rangle \sim \varepsilon^{-1/2} \to \infty$')
    ax.legend(fontsize=8, title='$\\varepsilon$ (distance)', title_fontsize=8)

    # Panel (b): Type-III cubic map — period-2, no tangency
    ax = axes[1]
    x2 = np.linspace(-1.5, 1.5, 500)

    # cubic drift: ẋ = ε x + A x³ → piecewise 2nd iterate
    eps_III_vals = [0.001, 0.005, 0.015, 0.04]
    cols_III = ['black', '#CC1100', '#EE4422', '#FF9966']

    for eps_III, col, lbl in zip(eps_III_vals, cols_III,
                                  ['$\\varepsilon=0.001$', '0.005', '0.015', '0.040']):
        # Sketch: ψ²(x) ≈ x + 2ε x + 2A x³ for illustration
        f2 = x2 + 2*eps_III*x2 + 2*A0*x2**3
        ax.plot(x2, f2, color=col, lw=1.5, label=lbl)

    ax.plot([-1.5, 1.5], [-1.5, 1.5], 'k-', lw=0.8, alpha=0.5)
    ax.axhspan(-0.1, 0.1, alpha=0.07, color='orange')
    ax.text(-1.35, 0.2, 'Period-2 orbit\nstraddles origin', fontsize=8, color='darkorange')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('$x_n$')
    ax.set_ylabel('$\\psi^2(x_n)$ (2nd iterate)')
    ax.set_title('(b) Type-III (period-doubling): cubic crossing at $\\varepsilon=0$\n'
                 r'No tangency; $l_\infty(u_0) = \frac{1}{2A}(u_0^{-2} - c^{-2}) < \infty$')
    ax.legend(fontsize=8, title='$\\varepsilon$', title_fontsize=8)

    fig.suptitle('Figure 5: Geometric reason — tangency vs crossing',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Figure 6: Summary — regimes of ε
# ---------------------------------------------------------------------------
def fig6_regime_diagram():
    s3 = load_step3()
    mask_th = s3['mean_l_th'] > 0

    fig, ax = plt.subplots(figsize=(9, 4.5))

    le  = np.log10(s3['eps_csv'][mask_th])
    lls = np.log10(s3['mean_l_sim'][mask_th])
    llt = np.log10(s3['mean_l_th'][mask_th])

    ax.plot(le, lls, 'o', color='#0055AA', ms=6, zorder=5, label='Simulation')
    ax.plot(le, llt, 's-', color='#CC2200', ms=4, lw=1.2, label='Theory')

    # Shade regimes
    e_arr = s3['eps_csv'][mask_th]
    # Saturation region: ~first 7 points
    e_sat_end = np.log10(e_arr[6])
    # Crossover region
    e_co_start = np.log10(e_arr[6])
    e_co_end   = np.log10(e_arr[-8])
    # Power-law region
    e_pl_start = np.log10(e_arr[-8])

    ax.axvspan(le.min()-0.1, e_sat_end, alpha=0.12, color='steelblue',
               label='Saturation regime ($\\gamma \\to 0$)')
    ax.axvspan(e_co_start, e_co_end, alpha=0.10, color='orange',
               label='Crossover ($\\gamma_{\\rm eff} \\approx 0.24$)')
    ax.axvspan(e_pl_start, le.max()+0.1, alpha=0.10, color='green',
               label='Theory accurate ($\\sim 1\\times$)')

    # Slope guides
    from scipy import stats as spstat
    mid = slice(8, -4)
    sl, ic, *_ = spstat.linregress(le[mid], lls[mid])
    x_fit = np.linspace(le[6], le[-5], 50)
    ax.plot(x_fit, sl*x_fit + ic, '--', color='#0055AA', lw=1.2, alpha=0.8,
            label=f'Fit: $\\gamma_{{\\rm eff}} = {-sl:.3f}$')

    # Annotate
    ax.text((le.min()+e_sat_end)/2, lls.max()-0.05,
            'Saturation\n$\\langle l\\rangle \\to$ const\n$\\gamma \\to 0$',
            ha='center', fontsize=9, color='steelblue',
            bbox=dict(fc='white', alpha=0.7, pad=2))
    ax.text((e_co_start+e_co_end)/2, lls.mean()-0.05,
            'Crossover\n"apparent"\n$\\gamma \\approx 0.24$',
            ha='center', fontsize=9, color='darkorange',
            bbox=dict(fc='white', alpha=0.7, pad=2))
    ax.text((e_pl_start+le.max())/2 + 0.05, lls.min()+0.05,
            'Large $\\varepsilon$:\ntheory accurate',
            ha='center', fontsize=9, color='darkgreen',
            bbox=dict(fc='white', alpha=0.7, pad=2))

    ax.set_xlabel(r'$\log_{10}\,\varepsilon$')
    ax.set_ylabel(r'$\log_{10}\,\langle l \rangle$')
    ax.set_title('Figure 6: Three regimes of $\\varepsilon$ — saturation, crossover, power-law')
    ax.legend(fontsize=8.5, loc='lower right')

    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Main: generate all figures
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    print("Generating Figure 1: Type-I vs Type-III conceptual comparison...")
    f1 = fig1_conceptual()
    f1.savefig(os.path.join(REPDIR, 'fig1_typeI_vs_typeIII.png'))
    f1.savefig(os.path.join(REPDIR, 'fig1_typeI_vs_typeIII.pdf'))
    plt.close(f1)
    print("  -> saved fig1_typeI_vs_typeIII.{png,pdf}")

    print("Generating Figure 2: Laminar length surface...")
    f2 = fig2_laminar_surface()
    f2.savefig(os.path.join(REPDIR, 'fig2_laminar_surface.png'))
    f2.savefig(os.path.join(REPDIR, 'fig2_laminar_surface.pdf'))
    f2.savefig(os.path.join(FIGDIR, 'fig2_laminar_surface.png'), dpi=200)
    f2.savefig(os.path.join(FIGDIR, 'fig2_laminar_surface.pdf'))
    plt.close(f2)
    print("  -> saved fig2_laminar_surface.{png,pdf}")

    print("Generating Figure 3: Saturation limit and RPD gap...")
    f3 = fig3_limit_formula()
    f3.savefig(os.path.join(REPDIR, 'fig3_limit_and_rpd_gap.png'))
    f3.savefig(os.path.join(REPDIR, 'fig3_limit_and_rpd_gap.pdf'))
    f3.savefig(os.path.join(FIGDIR, 'fig3_limit_and_rpd_gap.png'), dpi=200)
    f3.savefig(os.path.join(FIGDIR, 'fig3_limit_and_rpd_gap.pdf'))
    plt.close(f3)
    print("  -> saved fig3_limit_and_rpd_gap.{png,pdf}")

    print("Generating Figure 4: Data comparison...")
    f4 = fig4_data_comparison()
    f4.savefig(os.path.join(REPDIR, 'fig4_data_comparison.png'))
    f4.savefig(os.path.join(REPDIR, 'fig4_data_comparison.pdf'))
    plt.close(f4)
    print("  -> saved fig4_data_comparison.{png,pdf}")

    print("Generating Figure 5: Physical sketch...")
    f5 = fig5_physical_sketch()
    f5.savefig(os.path.join(REPDIR, 'fig5_physical_sketch.png'))
    f5.savefig(os.path.join(REPDIR, 'fig5_physical_sketch.pdf'))
    plt.close(f5)
    print("  -> saved fig5_physical_sketch.{png,pdf}")

    print("Generating Figure 6: Regime diagram...")
    f6 = fig6_regime_diagram()
    f6.savefig(os.path.join(REPDIR, 'fig6_regime_diagram.png'))
    f6.savefig(os.path.join(REPDIR, 'fig6_regime_diagram.pdf'))
    plt.close(f6)
    print("  -> saved fig6_regime_diagram.{png,pdf}")

    print("\nAll figures saved to:", REPDIR)
    print("Done.")
