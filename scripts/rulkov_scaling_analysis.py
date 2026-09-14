#!/usr/bin/env python3
"""
Rulkov 1D map: Scaling analysis for type-I intermittency.
- Load v2 numerical data (log10(eps) in [-10, -5])
- Compute theoretical <l> using piecewise RPD
- Plot log(eps) vs log(<l>) with slope annotations
- Assess theory-numerics discrepancy at small eps
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

DATADIR = os.path.join(os.path.dirname(__file__), '..', '..', 'code', 'datafiles')
OUTDIR = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11,
})

ALPHA_MAP = 4.8
GAMMA_C = -2.931400258306671
FIXED_POINT = -1.765458470442114

# ── Exact a2 from map curvature at tangent point ─────────────────────────
# F(x) = alpha/(1+x^2) + gamma
# F'(x) = -2*alpha*x/(1+x^2)^2
# F''(x) = -2*alpha*(1+x^2)^2 + 2*alpha*x*2*(1+x^2)*2*x / (1+x^2)^4
#        = -2*alpha*[(1+x^2) - 4x^2] / (1+x^2)^3
#        = -2*alpha*(1-3x^2) / (1+x^2)^3
# a2 = F''(fp)/2
fp = FIXED_POINT
A2_EXACT = -ALPHA_MAP * (1 - 3*fp**2) / (1 + fp**2)**3
print(f"Exact a2 from map curvature: {A2_EXACT:.8f}")


# ── RPD computation from reinjection data ────────────────────────────────
def M_function(x_sorted):
    return np.cumsum(x_sorted) / np.arange(1, len(x_sorted) + 1)


def compute_rpd_single_branch(x_sorted, c):
    M_vals = M_function(x_sorted)
    m, _ = np.polyfit(x_sorted, M_vals, 1)
    alpha_rpd = (2 * m - 1) / (1 - m)
    x_hat = x_sorted[0]
    norm = (c - x_hat)**(alpha_rpd + 1) / (alpha_rpd + 1) if alpha_rpd > -1 else (c - x_hat)
    b_phi = 1.0 / norm
    print(f"  Single-branch RPD: m={m:.6f}, alpha={alpha_rpd:.6f}, x_hat={x_hat:.6f}, b={b_phi:.6f}")
    return {'m': m, 'alpha': alpha_rpd, 'x_hat': x_hat, 'b': b_phi, 'c': c, 'n_branches': 1}


def phi_single(x, rpd):
    c = rpd['c']
    x_hat = rpd['x_hat']
    alpha = rpd['alpha']
    b = rpd['b']
    if x < x_hat or x > c:
        return 0.0
    return b * (x - x_hat)**alpha


# ── Laminar length for type-I ────────────────────────────────────────────
def laminar_length_type1(x_entry, c, a2, eps):
    if a2 * eps <= 0:
        return 0.0
    sqae = np.sqrt(a2 * eps)
    ratio = np.sqrt(a2 / eps)
    return (np.arctan(c * ratio) - np.arctan(x_entry * ratio)) / sqae


def mean_l_theoretical(c, a2, eps, rpd):
    def integrand(x):
        phi_val = phi_single(x, rpd)
        l_val = laminar_length_type1(x, c, a2, eps)
        return phi_val * max(l_val, 0.0)

    x_hat = rpd['x_hat']
    result, error = quad(integrand, x_hat, c, limit=500, epsabs=1e-12, epsrel=1e-10)
    return result


# ── Closed-form for c=0.1, uniform RPD (alpha=0) ────────────────────────
def mean_l_closed_form_uniform(c, a2, eps):
    """<l> for uniform RPD phi=1/(2c) over [-c, c], type-I."""
    sqae = np.sqrt(a2 * eps)
    ratio = np.sqrt(a2 / eps)
    atan_c = np.arctan(c * ratio)
    # <l> = (1/(2c)) * integral from -c to c of (1/sqae)[arctan(c*r) - arctan(x*r)] dx
    # = (1/(2c*sqae)) * [2c*arctan(c*r) - integral of arctan(x*r) dx from -c to c]
    # integral of arctan(a*x) dx = x*arctan(a*x) - ln(1+a^2*x^2)/(2a)
    # evaluated from -c to c: 2c*arctan(a*c) - ln(1+a^2*c^2)/a (since arctan is odd)
    # Wait, let me redo: integral from -c to c of arctan(r*x) dx
    # = [x*arctan(r*x)]_{-c}^c - integral_{-c}^c x * r/(1+r^2*x^2) dx
    # = 2c*arctan(r*c) - [ln(1+r^2*x^2)/(2)]_{-c}^c   (nope, wrong)
    # Let me use scipy instead.
    def integrand(x):
        return (np.arctan(c * ratio) - np.arctan(x * ratio)) / sqae / (2 * c)
    result, _ = quad(integrand, -c, c, limit=200)
    return result


# ── Process each case ────────────────────────────────────────────────────
def process_case(c_val):
    c_str = f"{c_val:.2f}"
    fname = os.path.join(DATADIR, f'rulkov_scaling_v2_c={c_str}.csv')
    reinj_file = os.path.join(DATADIR, f'rulkov_reinjection_c={c_str}.dat')

    if not os.path.exists(fname):
        print(f"Data file not found: {fname}")
        return None

    data = np.loadtxt(fname, delimiter=',', comments='#')
    log10_eps = data[:, 0]
    eps = data[:, 1]
    mean_l = data[:, 3]
    max_l = data[:, 4]
    std_l = data[:, 5]
    n_episodes = data[:, 6].astype(int)
    a2_fit = data[:, 7]
    a1_fit = data[:, 8]

    mask = (mean_l > 0) & (n_episodes >= 100)
    eps_m = eps[mask]
    log10_eps_m = log10_eps[mask]
    mean_l_m = mean_l[mask]
    max_l_m = max_l[mask]
    std_l_m = std_l[mask]
    a2_m = a2_fit[mask]

    print(f"\n{'='*60}")
    print(f"Case: c = {c_val}")
    print(f"{'='*60}")
    print(f"Points: {np.sum(mask)}/{len(mask)}")
    print(f"eps range: [{eps_m.min():.4e}, {eps_m.max():.4e}]")
    print(f"log10(eps) range: [{log10_eps_m.min():.2f}, {log10_eps_m.max():.2f}]")
    print(f"a2 range: [{a2_m.min():.6f}, {a2_m.max():.6f}]")
    print(f"Exact a2: {A2_EXACT:.6f}")

    # RPD from reinjection data
    rpd = None
    if os.path.exists(reinj_file):
        print(f"\nComputing RPD from reinjection data...")
        reinj_data = np.loadtxt(reinj_file, comments='#')
        x_all = reinj_data.ravel()
        # Filter to [-c, c]
        x_in = x_all[(x_all >= -c_val) & (x_all <= c_val)]
        x_sorted = np.sort(x_in)
        if len(x_sorted) > 20000:
            rng = np.random.default_rng(42)
            x_sorted = np.sort(rng.choice(x_sorted, 20000, replace=False))
        rpd = compute_rpd_single_branch(x_sorted, c_val)
    else:
        print("No reinjection data. Using uniform RPD (alpha=0).")
        rpd = {'m': 0.5, 'alpha': 0.0, 'x_hat': -c_val, 'b': 1.0 / (2 * c_val),
               'c': c_val, 'n_branches': 1}

    # Verify RPD normalization
    norm_check, _ = quad(lambda x: phi_single(x, rpd), rpd['x_hat'], c_val, limit=200)
    print(f"  RPD normalization: {norm_check:.6f} (should be 1.0)")

    # Theoretical <l>
    print(f"\nComputing theoretical <l> at each eps...")
    mean_l_th = np.zeros_like(eps_m)
    for i in range(len(eps_m)):
        mean_l_th[i] = mean_l_theoretical(c_val, A2_EXACT, eps_m[i], rpd)

    # Also compute with uniform RPD for comparison
    rpd_uniform = {'m': 0.5, 'alpha': 0.0, 'x_hat': -c_val, 'b': 1.0 / (2 * c_val),
                   'c': c_val, 'n_branches': 1}
    mean_l_th_uniform = np.zeros_like(eps_m)
    for i in range(len(eps_m)):
        mean_l_th_uniform[i] = mean_l_theoretical(c_val, A2_EXACT, eps_m[i], rpd_uniform)

    # Power law fits: log(mean_l) = -nu * log(eps) + log(K)
    log_eps = np.log(eps_m)
    log_ml = np.log(mean_l_m)

    coeffs_num, cov_num = np.polyfit(log_eps, log_ml, 1, cov=True)
    nu_num = -coeffs_num[0]
    nu_num_err = np.sqrt(cov_num[0, 0])

    mask_th = mean_l_th > 0
    if np.sum(mask_th) > 3:
        coeffs_th, cov_th = np.polyfit(np.log(eps_m[mask_th]), np.log(mean_l_th[mask_th]), 1, cov=True)
        nu_th = -coeffs_th[0]
        nu_th_err = np.sqrt(cov_th[0, 0])
    else:
        nu_th = nu_th_err = np.nan

    # Local slopes (sliding window)
    window = 5
    local_nu = np.full(len(eps_m), np.nan)
    for i in range(window, len(eps_m) - window):
        sl = slice(i - window, i + window + 1)
        c_loc, _ = np.polyfit(log_eps[sl], log_ml[sl], 1, cov=False)
        local_nu[i] = -c_loc

    # Ratio theory/numerics
    valid = mean_l_th > 0
    ratio = mean_l_th[valid] / mean_l_m[valid]

    print(f"\nSCALING RESULTS:")
    print(f"  nu (numerical):    {nu_num:.4f} +/- {nu_num_err:.4f}")
    print(f"  nu (theoretical):  {nu_th:.4f} +/- {nu_th_err:.4f}")
    print(f"  Expected (type-I): 0.5000")
    print(f"  Theory/numerics ratio: [{ratio.min():.3f}, {ratio.max():.3f}]")
    print(f"  Mean ratio: {ratio.mean():.3f}")

    return {
        'c': c_val,
        'eps': eps_m, 'log10_eps': log10_eps_m,
        'mean_l': mean_l_m, 'max_l': max_l_m, 'std_l': std_l_m,
        'mean_l_th': mean_l_th, 'mean_l_th_uniform': mean_l_th_uniform,
        'nu_num': nu_num, 'nu_num_err': nu_num_err,
        'nu_th': nu_th, 'nu_th_err': nu_th_err,
        'local_nu': local_nu,
        'rpd': rpd, 'ratio': ratio,
    }


# ── Main ─────────────────────────────────────────────────────────────────
results = {}
for c_val in [0.1, 0.01]:
    r = process_case(c_val)
    if r is not None:
        results[c_val] = r

# ── Plotting ─────────────────────────────────────────────────────────────
TICK_SIZE = 18
LABEL_SIZE = 22

def set_tick_params(ax, nticks=4):
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.locator_params(axis='x', nbins=nticks)
    ax.locator_params(axis='y', nbins=nticks)


for c_val, r in results.items():
    c_str = f"{c_val:.2f}"

    log_eps = np.log(r['eps'])
    log_ml = np.log(r['mean_l'])
    log_ml_th = np.log(r['mean_l_th'])
    log_ml_max = np.log(r['max_l'])

    # ── Main scaling plot: log(eps) vs log(<l>) ──────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(log_eps, log_ml, 'bo', ms=5, label=f'Numerical ($\\nu={r["nu_num"]:.3f}\\pm{r["nu_num_err"]:.3f}$)')

    mask_th = r['mean_l_th'] > 0
    if np.any(mask_th):
        ax.plot(log_eps[mask_th], log_ml_th[mask_th], 'r^', ms=4, alpha=0.7,
                label=f'Theoretical ($\\nu={r["nu_th"]:.3f}\\pm{r["nu_th_err"]:.3f}$)')

    # Fit line
    fit_x = np.linspace(log_eps.min(), log_eps.max(), 100)
    fit_y = -r['nu_num'] * fit_x + np.log(r['mean_l'].max() * r['eps'].max()**r['nu_num'])
    # Recompute intercept
    c_fit = np.polyfit(log_eps, log_ml, 1)
    fit_y = np.polyval(c_fit, fit_x)
    ax.plot(fit_x, fit_y, 'b--', lw=1, alpha=0.4)

    # Reference nu=0.5 line
    c_ref = np.polyfit(log_eps, log_ml, 1)
    mid_eps = np.median(log_eps)
    mid_ml = np.polyval(c_ref, mid_eps)
    ref_y = -0.5 * (fit_x - mid_eps) + mid_ml
    ax.plot(fit_x, ref_y, 'k:', lw=1, alpha=0.3, label=r'Reference $\nu=0.5$')

    ax.set_xlabel(r'$\ln(\varepsilon)$', fontsize=LABEL_SIZE)
    ax.set_ylabel(r'$\ln(\langle l \rangle)$', fontsize=LABEL_SIZE)
    ax.set_title(f'Rulkov type-I: $c = {c_str}$', fontsize=16)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    set_tick_params(ax)

    # Slope annotation
    ax.text(0.05, 0.05, f'slope $= -{r["nu_num"]:.4f}$',
            transform=ax.transAxes, fontsize=12, va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f'rulkov_scaling_v2_c={c_str}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUTDIR, f'rulkov_scaling_v2_c={c_str}.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: rulkov_scaling_v2_c={c_str}.png/pdf")

    # ── Mean + max scaling plot ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(log_eps, log_ml, 'bo', ms=5, label=r'$\langle l \rangle$ (numerical)')
    ax.plot(log_eps, log_ml_max, 'gs', ms=4, alpha=0.6, label=r'$l_{\max}$ (numerical)')
    if np.any(mask_th):
        ax.plot(log_eps[mask_th], log_ml_th[mask_th], 'r-', lw=1.5, alpha=0.7, label=r'$\langle l \rangle$ (theoretical)')

    # Fit l_max
    c_max = np.polyfit(log_eps, log_ml_max, 1)
    nu_max = -c_max[0]
    ax.plot(fit_x, np.polyval(c_max, fit_x), 'g--', lw=1, alpha=0.4)

    ax.set_xlabel(r'$\ln(\varepsilon)$', fontsize=LABEL_SIZE)
    ax.set_ylabel(r'$\ln(l)$', fontsize=LABEL_SIZE)
    ax.set_title(f'Rulkov type-I: $c = {c_str}$ — Mean and max', fontsize=16)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    set_tick_params(ax)
    ax.text(0.05, 0.15, f'$\\nu_{{\\langle l \\rangle}} = {r["nu_num"]:.4f}$\n$\\nu_{{l_{{\\max}}}} = {nu_max:.4f}$',
            transform=ax.transAxes, fontsize=12, va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f'rulkov_scaling_v2_c={c_str}_mean_max.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: rulkov_scaling_v2_c={c_str}_mean_max.png")

    # ── Local slope plot ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    valid_nu = ~np.isnan(r['local_nu'])
    ax.plot(log_eps[valid_nu], r['local_nu'][valid_nu], 'ko-', ms=4, lw=1)
    ax.axhline(0.5, color='r', ls='--', lw=1, alpha=0.5, label=r'$\nu=0.5$ (classical)')
    ax.axhline(r['nu_num'], color='b', ls=':', lw=1, alpha=0.5, label=f'Global fit $\\nu={r["nu_num"]:.3f}$')
    ax.set_xlabel(r'$\ln(\varepsilon)$', fontsize=LABEL_SIZE)
    ax.set_ylabel(r'$\nu_{\mathrm{local}}$', fontsize=LABEL_SIZE)
    ax.set_title(f'Local scaling exponent, $c = {c_str}$', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    set_tick_params(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f'rulkov_local_slope_c={c_str}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: rulkov_local_slope_c={c_str}.png")

    # ── Theory/numerics ratio ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(log_eps[mask_th], r['ratio'], 'ko-', ms=4, lw=1)
    ax.axhline(1.0, color='r', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel(r'$\ln(\varepsilon)$', fontsize=LABEL_SIZE)
    ax.set_ylabel(r'$\langle l \rangle_{\mathrm{th}} / \langle l \rangle_{\mathrm{num}}$', fontsize=LABEL_SIZE)
    ax.set_title(f'Theory / numerics ratio, $c = {c_str}$', fontsize=14)
    ax.grid(True, alpha=0.2)
    set_tick_params(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f'rulkov_ratio_c={c_str}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: rulkov_ratio_c={c_str}.png")


# ── Combined figure ──────────────────────────────────────────────────────
if len(results) == 2:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, c_val, panel in zip([ax1, ax2], [0.1, 0.01], ['(a)', '(b)']):
        r = results[c_val]
        log_eps = np.log(r['eps'])
        log_ml = np.log(r['mean_l'])
        log_ml_th = np.log(r['mean_l_th'])

        ax.plot(log_eps, log_ml, 'bo', ms=5,
                label=f'Numerical ($\\nu={r["nu_num"]:.3f}$)')

        mask_th = r['mean_l_th'] > 0
        if np.any(mask_th):
            ax.plot(log_eps[mask_th], log_ml_th[mask_th], 'r-', lw=1.5, alpha=0.7,
                    label=f'Theoretical ($\\nu={r["nu_th"]:.3f}$)')

        fit_x = np.linspace(log_eps.min(), log_eps.max(), 100)
        c_fit = np.polyfit(log_eps, log_ml, 1)
        ax.plot(fit_x, np.polyval(c_fit, fit_x), 'b--', lw=1, alpha=0.3)

        ax.set_xlabel(r'$\ln(\varepsilon)$', fontsize=LABEL_SIZE)
        ax.set_ylabel(r'$\ln(\langle l \rangle)$', fontsize=LABEL_SIZE)
        ax.set_title(f'{panel} $c = {c_val}$', loc='left', fontsize=16)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        set_tick_params(ax)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'rulkov_scaling_v2_combined.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUTDIR, 'rulkov_scaling_v2_combined.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("Saved: rulkov_scaling_v2_combined.png/pdf")

# ── Summary table ────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for c_val, r in results.items():
    print(f"\nc = {c_val}:")
    print(f"  nu_num = {r['nu_num']:.4f} +/- {r['nu_num_err']:.4f}")
    print(f"  nu_th  = {r['nu_th']:.4f} +/- {r['nu_th_err']:.4f}")
    print(f"  RPD alpha = {r['rpd']['alpha']:.4f}")
    print(f"  Theory/num ratio: mean={r['ratio'].mean():.3f}, range=[{r['ratio'].min():.3f}, {r['ratio'].max():.3f}]")
    if r['ratio'].mean() < 1.0:
        print(f"  ** Theory UNDERESTIMATES by factor {1/r['ratio'].mean():.2f} on average")
