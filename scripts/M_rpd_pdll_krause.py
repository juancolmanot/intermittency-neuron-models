#!/usr/bin/env python3
"""
Unified M / RPD / PDLL: 15 rows × 3 cols.

Implements Krause et al. 2014 method properly:
  - Branch 1 (inverted): φ_1(x) = b·(hi - x)^α_1
  - Branch j (normal):   φ_j(x) = b·k_j·(x - lo_j)^α_j
  - k_j determined by M continuity at breakpoints
  - b from normalization ∫φ=1

M_j built cumulatively: integral definition → N_j/D_j.
Padding: fits exclude PAD from each non-boundary edge.

Col 0: M(u) theo vs num
Col 1: RPD φ(u) theo vs num
Col 2: PDLL ψ(l) theo vs num (pushforward through l_cont)
"""
import os, numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(BASE, 'data')
FIG  = os.path.join(BASE, 'figures')

DVALS = ['-11.795',
         '-11.80', '-11.81', '-11.82', '-11.83', '-11.84',
         '-11.86', '-11.88', '-11.90', '-11.94', '-11.96', '-11.98',
         '-12.00', '-12.02', '-12.04', '-12.06', '-12.08', '-12.10',
         '-12.12', '-12.14', '-12.16', '-12.18', '-12.20', '-12.25',
         '-12.30', '-12.35', '-12.45', '-12.50', '-12.55',
         '-12.60', '-12.65', '-12.80']

_SLOPE_DEFAULTS_200 = {
    '-11.795': [-0.665],
    '-11.80': [-0.667],
    '-11.81': [-0.672],
    '-11.82': [-0.678],
    '-11.83': [-0.682],
    '-11.84': [-0.687],
    '-11.86': [-0.697],
    '-11.88': [-0.707],
    '-11.90': [-0.676],
    '-11.94': [-0.721],
    '-11.96': [-0.748],
    '-11.98': [-0.743],
    '-12.00': [-0.799],
    '-12.02': [-0.778],
    '-12.04': [-0.788],
    '-12.06': [-0.799],
    '-12.08': [-0.809],
    '-12.10': [-0.815],
    '-12.12': [-0.829],
    '-12.14': [-0.840],
    '-12.16': [-0.850],
    '-12.18': [-0.860],
    '-12.20': [-0.871],
    '-12.25': [-0.896],
    '-12.30': [-0.922],
    '-12.35': [-0.948],
    '-12.45': [-1.001, 1.15],
    '-12.50': [-1.48, -1.28, -1.03],
    '-12.55': [-1.054],
    '-12.60': [-1.080, 1.28],
    '-12.65': [-1.110],
    '-12.80': [-1.188, 0.80],
}

BKPTS = {d: _SLOPE_DEFAULTS_200[d] for d in DVALS}

# Per-branch inversion overrides: d -> {branch_index: True/False}
# Default: only branch 0 is inverted
INV_OVERRIDES = {
    '-12.50': {2: True},
    '-12.80': {1: True},
}

PAD = 0.05
PAD_INNER_FRAC = 0.25

# Per-branch padding overrides: d -> {branch_index: pad_frac}
PAD_OVERRIDES = {
    '-12.45': {2: 0.02},
    '-12.50': {2: 0.05},
}
FIT_W = 0.5
HEAD_L = 15

C_NUM  = '#1f5fbf'
C_THEO = '#c0392b'


# ── data loading ──────────────────────────────────────────────────────

def load_pts(d, clam):
    meta = np.loadtxt(os.path.join(DATA, f'd{d}_meta.txt'))
    eps = meta[1]
    ep = np.loadtxt(os.path.join(DATA, f'd{d}_ep_clam{clam:.2f}.dat'), comments='#')
    u0 = ep[:, 0]
    return eps, u0[(u0 > -clam) & (u0 < clam)]


def load_laminar(d, clam):
    ep = np.loadtxt(os.path.join(DATA, f'd{d}_ep_clam{clam:.2f}.dat'), comments='#')
    u_reinj = ep[:, 0]
    l_sim = ep[:, 2]
    mask = (np.abs(u_reinj) < clam) & (np.abs(u_reinj) > 1e-6)
    return u_reinj[mask], l_sim[mask]


def fit_A(d):
    dat = np.loadtxt(os.path.join(DATA, f'd{d}_mapcurve.dat'), comments='#')
    s, y = dat[:, 0], dat[:, 1]
    m = np.abs(s) < FIT_W
    if m.sum() < 10:
        m = np.abs(s) < 1.0
    M = np.column_stack([s[m], s[m]**3])
    (C, A), *_ = np.linalg.lstsq(M, y[m], rcond=None)
    return C, A


# ── branch fitting ───────────────────────────────────────────────────

def alpha_from_slope(m, inverted=False):
    if inverted:
        return (-2*m - 1) / (1 + m) if abs(1 + m) > 1e-10 else np.nan
    return (2*m - 1) / (1 - m) if abs(1 - m) > 1e-10 else np.nan


def fit_branch(pts, lo, hi, inverted=False, is_lo_boundary=False, is_hi_boundary=False, pad=None):
    if pad is None:
        pad = PAD
    bp = pts[(pts > lo) & (pts < hi)] if inverted else pts[(pts >= lo) & (pts < hi)]
    if len(bp) < 20:
        return None
    if inverted:
        bp_s = np.sort(bp)[::-1]
        refl = hi + (hi - bp_s)
        M_arr = np.cumsum(refl) / np.arange(1, len(refl)+1, dtype=float)
    else:
        bp_s = np.sort(bp)
        M_arr = np.cumsum(bp_s) / np.arange(1, len(bp_s)+1, dtype=float)
    fit_lo = lo if is_lo_boundary else lo + pad
    fit_hi = hi if is_hi_boundary else hi - pad
    fit_mask = (bp_s >= fit_lo) & (bp_s <= fit_hi)
    if fit_mask.sum() < 20:
        fit_mask = np.ones(len(bp_s), dtype=bool)
    s = stats.linregress(bp_s[fit_mask], M_arr[fit_mask])
    # Clean slope: M computed from fit-region points only (no peak tail contamination)
    fit_pts = bp_s[fit_mask]
    if inverted:
        fit_refl = hi + (hi - fit_pts)
        M_clean = np.cumsum(fit_refl) / np.arange(1, len(fit_refl)+1, dtype=float)
    else:
        M_clean = np.cumsum(fit_pts) / np.arange(1, len(fit_pts)+1, dtype=float)
    s_clean = stats.linregress(fit_pts, M_clean)
    return dict(u=bp_s, M=M_arr, m=s.slope, b_int=s.intercept,
                R2=s.rvalue**2, n=len(bp), inverted=inverted,
                lo=lo, hi=hi,
                m_clean=s_clean.slope, b_int_clean=s_clean.intercept,
                R2_clean=s_clean.rvalue**2,
                fit_lo=fit_lo, fit_hi=fit_hi)


def get_branches(pts, clam, bkpts, inv_override=None, pad_override=None):
    edges = [-clam] + list(bkpts) + [clam]
    brs = []
    for i in range(len(edges)-1):
        if inv_override and i in inv_override:
            inv = inv_override[i]
        else:
            inv = (i == 0)
        branch_width = edges[i+1] - edges[i]
        if pad_override and i in pad_override:
            inner_pad = pad_override[i] * branch_width
        else:
            inner_pad = PAD_INNER_FRAC * branch_width if i > 0 else PAD
        br = fit_branch(pts, edges[i], edges[i+1], inverted=inv,
                        is_lo_boundary=(i == 0),
                        is_hi_boundary=(i == len(edges)-2),
                        pad=inner_pad)
        brs.append(br)
    return edges, brs


# ── Krause theoretical M / RPD ───────────────────────────────────────

def krause_theory(branches, edges, clam, ngrid=8000):
    """
    Build theoretical M and RPD using Krause cumulative integral method.

    B1 (inverted): φ_1 = b·(hi_1 - x)^α_1
    Bj (j≥2):     φ_j = b·k_j·(x - lo_j)^α_j

    k_j from M continuity at breakpoints.
    b from normalization.
    """
    nbr = len(branches)
    alphas = []
    for j, br in enumerate(branches):
        if br is None:
            alphas.append(0.0)
        else:
            m_use = br.get('m_clean', br['m'])
            alphas.append(alpha_from_slope(m_use, br['inverted']))

    # Helpers for power-law integrals
    def N_integral(lo, hi, alpha, ref, inverted):
        """∫_lo^hi τ·(dist)^α dτ where dist = ref-τ (inv) or τ-ref (normal)"""
        L = hi - lo
        if inverted:
            # dist = ref - τ, with ref = hi of branch
            # ∫ τ·(ref-τ)^α dτ from lo to hi
            # = ref·L^(α+1)/(α+1) - L^(α+2)/(α+2)  [change of var t = ref-τ]
            # wait — more careful: let t = ref - τ, dτ = -dt
            # ∫_lo^hi τ·(ref-τ)^α dτ = ∫_{ref-hi}^{ref-lo} (ref-t)·t^α dt
            # = ref·[(ref-lo)^(α+1) - (ref-hi)^(α+1)]/(α+1)
            #   - [(ref-lo)^(α+2) - (ref-hi)^(α+2)]/(α+2)
            t_lo = ref - hi  # = 0 when ref = hi
            t_hi = ref - lo
            return (ref * (t_hi**(alpha+1) - t_lo**(alpha+1)) / (alpha+1)
                    - (t_hi**(alpha+2) - t_lo**(alpha+2)) / (alpha+2))
        else:
            # dist = τ - ref, ref = lo of branch
            # ∫_lo^hi τ·(τ-ref)^α dτ = ∫_0^{hi-ref} (ref+t)·t^α dt
            t_hi = hi - ref
            t_lo = lo - ref
            t_lo = max(t_lo, 0)
            return (ref * (t_hi**(alpha+1) - t_lo**(alpha+1)) / (alpha+1)
                    + (t_hi**(alpha+2) - t_lo**(alpha+2)) / (alpha+2))

    def D_integral(lo, hi, alpha, ref, inverted):
        """∫_lo^hi (dist)^α dτ"""
        if inverted:
            t_lo = ref - hi
            t_hi = ref - lo
            return (t_hi**(alpha+1) - t_lo**(alpha+1)) / (alpha+1)
        else:
            t_hi = hi - ref
            t_lo = lo - ref
            t_lo = max(t_lo, 0)
            return (t_hi**(alpha+1) - t_lo**(alpha+1)) / (alpha+1)

    # For each branch j, define ref and inverted flag
    refs = []
    invs = []
    for j, br in enumerate(branches):
        if br is None:
            refs.append(edges[j])
            invs.append(False)
        elif br['inverted']:
            refs.append(edges[j+1])  # upper breakpoint
            invs.append(True)
        else:
            refs.append(edges[j])    # lower breakpoint
            invs.append(False)

    # Numerical M for matching at breakpoints
    all_pts_sorted = []
    for br in branches:
        if br:
            all_pts_sorted.extend(br['u'].tolist())
    all_pts_sorted = np.sort(np.array(all_pts_sorted))
    M_num_full = np.cumsum(all_pts_sorted) / np.arange(1, len(all_pts_sorted)+1, dtype=float)

    def M_num_at(x_val):
        idx = np.searchsorted(all_pts_sorted, x_val)
        idx = min(idx, len(M_num_full)-1)
        return M_num_full[idx]

    # Determine k_j via Krause method (Eq 14):
    # k[0] = 1. For j>=1, solve M_theo(edge[j+1]) = M_num(edge[j+1]).
    # Calibration: at each internal breakpoint, anchor cum_N to
    # M_num × cum_D so the global M matches data there. This follows
    # Krause's approach of using numerical M(x) in the k equation.
    ks = np.ones(nbr)

    # Precompute all branch integrals
    Ns = np.zeros(nbr)
    Ds = np.zeros(nbr)
    for j in range(nbr):
        if branches[j] is None:
            continue
        Ns[j] = N_integral(edges[j], edges[j+1], alphas[j], refs[j], invs[j])
        Ds[j] = D_integral(edges[j], edges[j+1], alphas[j], refs[j], invs[j])

    cum_N_at_edge = np.zeros(nbr + 1)
    cum_D_at_edge = np.zeros(nbr + 1)

    for j in range(nbr):
        if branches[j] is None:
            cum_N_at_edge[j+1] = cum_N_at_edge[j]
            cum_D_at_edge[j+1] = cum_D_at_edge[j]
            continue

        if j > 0:
            M_target = M_num_at(edges[j+1])
            num = cum_N_at_edge[j] - M_target * cum_D_at_edge[j]
            den = M_target * Ds[j] - Ns[j]
            if abs(den) > 1e-15:
                ks[j] = num / den
            if ks[j] < 0:
                ks[j] = 0.0

        cum_N_at_edge[j+1] = cum_N_at_edge[j] + ks[j] * Ns[j]
        cum_D_at_edge[j+1] = cum_D_at_edge[j] + ks[j] * Ds[j]

        # Anchor cum_N at internal breakpoints to numerical M so that
        # the k equation for the NEXT branch uses the correct starting M.
        if j < nbr - 1 and cum_D_at_edge[j+1] > 1e-15:
            cum_N_at_edge[j+1] = M_num_at(edges[j+1]) * cum_D_at_edge[j+1]

    # Normalization: b = 1 / sum(k_j · D_j)
    b_norm = 1.0 / cum_D_at_edge[nbr] if cum_D_at_edge[nbr] > 1e-15 else 1.0

    # Build on fine grid
    x = np.linspace(-clam, clam, ngrid)
    phi = np.zeros(ngrid)
    cum_N_grid = np.zeros(ngrid)
    cum_D_grid = np.zeros(ngrid)
    M_theo = np.zeros(ngrid)

    for j in range(nbr):
        if branches[j] is None:
            continue
        lo_j, hi_j = edges[j], edges[j+1]
        alpha_j = alphas[j]
        mask = (x >= lo_j) & (x <= hi_j)

        for ii in np.where(mask)[0]:
            xi = x[ii]
            if invs[j]:
                dist = refs[j] - xi
            else:
                dist = xi - refs[j]
            dist = max(dist, 1e-15)
            phi[ii] = b_norm * ks[j] * dist**alpha_j

            Nj_partial = N_integral(lo_j, xi, alpha_j, refs[j], invs[j])
            Dj_partial = D_integral(lo_j, xi, alpha_j, refs[j], invs[j])
            cum_N_grid[ii] = cum_N_at_edge[j] + ks[j] * Nj_partial
            cum_D_grid[ii] = cum_D_at_edge[j] + ks[j] * Dj_partial

    # M = cumN / cumD — suppress near-boundary where cum_D is tiny
    D_max = cum_D_grid.max()
    valid = cum_D_grid > 0.01 * D_max
    M_theo[:] = np.nan
    M_theo[valid] = cum_N_grid[valid] / cum_D_grid[valid]

    return x, phi, M_theo, ks, b_norm, alphas


# ── PDLL pushforward ─────────────────────────────────────────────────

def l_cont_func(u, eps, A, c):
    u2 = np.clip(u**2, 1e-8, None)
    arg = c**2 * (eps + A * u2) / (u2 * (eps + A * c**2))
    return np.log(np.clip(arg, 1.0, None)) / (2 * eps)


def pdll_pmf(lvals, lmax):
    li = np.clip(np.rint(lvals).astype(int), 1, lmax)
    counts = np.bincount(li, minlength=lmax + 1)[1:lmax + 1]
    s = counts.sum()
    P = counts / s if s > 0 else counts.astype(float)
    return np.arange(1, lmax + 1), P


# ── main ──────────────────────────────────────────────────────────────

def run(clam=2.0):
    tag = f'{clam:.2f}'.replace('.', '')
    print(f'\n{"="*70}')
    print(f'  Krause M/RPD/PDLL — clam={clam}')
    print(f'{"="*70}')

    records = []
    for d in DVALS:
        eps, pts = load_pts(d, clam)
        bkpts = BKPTS[d]
        edges, branches = get_branches(pts, clam, bkpts,
                                       inv_override=INV_OVERRIDES.get(d),
                                       pad_override=PAD_OVERRIDES.get(d))
        nbr = len(bkpts) + 1

        # Numerical M
        u_s = np.sort(pts)
        M_s = np.cumsum(u_s) / np.arange(1, len(u_s)+1, dtype=float)

        # Numerical RPD
        nbins = 200
        counts, rpd_edges = np.histogram(pts, bins=nbins, range=(-clam, clam))
        rpd_c = 0.5 * (rpd_edges[:-1] + rpd_edges[1:])
        rpd_d = counts / (counts.sum() * (rpd_edges[1] - rpd_edges[0]))

        # Theoretical M, RPD
        x_theo, phi_theo, M_theo, ks, b_norm, alphas = krause_theory(
            branches, edges, clam)

        # PDLL
        u_reinj, l_sim = load_laminar(d, clam)
        C_coeff, A_coeff = fit_A(d)
        eps_local = C_coeff - 1.0
        l_theo = np.maximum(1, np.ceil(l_cont_func(u_reinj, eps_local, A_coeff, clam)))
        lmax = max(int(l_sim.max()), int(l_theo.max()))
        l_arr, psi_num = pdll_pmf(l_sim, lmax)
        _, psi_theo = pdll_pmf(l_theo, lmax)
        lmax_plot = int(np.percentile(l_sim, 99.5)) + 1

        a_str = ', '.join(f'{a:.3f}' for a in alphas if not np.isnan(a))
        k_str = ', '.join(f'{k:.3f}' for k in ks)
        print(f'd={d}  {nbr}br  α=[{a_str}]  k=[{k_str}]  b={b_norm:.4f}')

        records.append(dict(
            d=d, eps=eps, nbr=nbr, bkpts=list(bkpts),
            u_s=u_s, M_s=M_s, rpd_c=rpd_c, rpd_d=rpd_d,
            x_theo=x_theo, phi_theo=phi_theo, M_theo=M_theo,
            alphas=alphas, ks=ks, b_norm=b_norm,
            l_arr=l_arr, psi_num=psi_num, psi_theo=psi_theo,
            lmax_plot=lmax_plot,
            mean_l_sim=l_sim.mean(), mean_l_theo=l_theo.mean(),
        ))

    # ── Figure ────────────────────────────────────────────────────────
    nrows = len(records)
    fig, axes = plt.subplots(nrows, 3, figsize=(18, 3.0 * nrows),
                              gridspec_kw={'width_ratios': [1.2, 1.2, 1.0]})
    bk_colors = ['#c0392b', '#2980b9', '#27ae60', '#e67e22', '#8e44ad']

    for ri, rec in enumerate(records):
        # ── Col 0: M-function ─────────────────────────────────────────
        ax0 = axes[ri, 0]
        step = max(1, len(rec['u_s']) // 4000)
        idx = np.arange(0, len(rec['u_s']), step)
        ax0.plot(rec['u_s'][idx], rec['M_s'][idx], '.', color=C_NUM,
                 ms=0.4, alpha=0.3, rasterized=True, label='numerical')
        ax0.plot(rec['x_theo'], rec['M_theo'], '-', color=C_THEO,
                 lw=1.2, alpha=0.85, label='theory')
        for bi, bk in enumerate(rec['bkpts']):
            ax0.axvline(bk, color=bk_colors[bi % len(bk_colors)],
                        ls='--', lw=0.7, alpha=0.5)
        m_min, m_max = rec['M_s'][0], rec['M_s'][-1]
        m_range = m_max - m_min
        ax0.set_ylim(m_min - 0.05*m_range, m_max + 0.05*m_range)
        ax0.set_ylabel(f'd={rec["d"]}', fontsize=8)
        ax0.tick_params(labelsize=6)
        a_str = ', '.join(f'α{j+1}={a:.2f}' for j, a in enumerate(rec['alphas'])
                          if not np.isnan(a))
        ax0.text(0.02, 0.95, a_str, transform=ax0.transAxes, fontsize=5.5,
                 va='top', family='monospace',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
        if ri == 0:
            ax0.set_title('M(u) — numerical vs Krause theory', fontsize=10,
                          fontweight='bold')
            ax0.legend(fontsize=7, loc='lower right')

        # ── Col 1: RPD ────────────────────────────────────────────────
        ax1 = axes[ri, 1]
        ax1.bar(rec['rpd_c'], rec['rpd_d'], width=(2*clam/200)*0.9,
                color='#bdc3c7', alpha=0.7, edgecolor='none', label='numerical')
        phi_clip = np.clip(rec['phi_theo'], 0,
                           rec['rpd_d'].max() * 3 if rec['rpd_d'].max() > 0 else 10)
        ax1.plot(rec['x_theo'], phi_clip, '-', color=C_THEO, lw=1.0,
                 alpha=0.85, label='theory')
        for bi, bk in enumerate(rec['bkpts']):
            ax1.axvline(bk, color=bk_colors[bi % len(bk_colors)],
                        ls='--', lw=0.7, alpha=0.5)
        ax1.set_xlim(-clam, clam)
        ax1.set_ylim(0, rec['rpd_d'].max() * 1.15)
        ax1.tick_params(labelsize=6)
        if ri == 0:
            ax1.set_title('RPD φ(u) — numerical vs theory', fontsize=10,
                          fontweight='bold')
            ax1.legend(fontsize=7, loc='upper right')

        # ── Col 2: PDLL ───────────────────────────────────────────────
        ax2 = axes[ri, 2]
        lmax_p = rec['lmax_plot']
        m_n = rec['psi_num'] > 0
        m_t = rec['psi_theo'] > 0
        clip = rec['l_arr'] <= lmax_p
        ax2.scatter(rec['l_arr'][m_n & clip], rec['psi_num'][m_n & clip],
                    s=6, color=C_NUM, alpha=0.6, label='num', zorder=3)
        ax2.scatter(rec['l_arr'][m_t & clip], rec['psi_theo'][m_t & clip],
                    s=6, color=C_THEO, marker='s', alpha=0.5, label='theo', zorder=2)
        ax2.set_yscale('log')
        pmin = min(rec['psi_num'][m_n].min() if m_n.any() else 1,
                   rec['psi_theo'][m_t].min() if m_t.any() else 1) * 0.3
        ax2.set_ylim(max(pmin, 1e-6), None)
        ax2.set_xlim(0, lmax_p)
        ax2.tick_params(labelsize=6)
        ratio = rec['mean_l_theo'] / rec['mean_l_sim'] if rec['mean_l_sim'] > 0 else 0
        ax2.text(0.97, 0.95,
                 f'⟨l⟩sim={rec["mean_l_sim"]:.1f}\n'
                 f'⟨l⟩theo={rec["mean_l_theo"]:.1f}\n'
                 f'ratio={ratio:.3f}',
                 transform=ax2.transAxes, fontsize=5.5, va='top', ha='right',
                 family='monospace',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
        if ri == 0:
            ax2.set_title('PDLL ψ(l) — numerical vs theory', fontsize=10,
                          fontweight='bold')
            ax2.legend(fontsize=7, loc='upper center')

    fig.suptitle(f'Krause M / RPD / PDLL — clam={clam}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    outf = os.path.join(FIG, f'krause_M_rpd_pdll_clam{tag}.png')
    fig.savefig(outf, dpi=150)
    print(f'\nWrote {outf}')
    plt.close(fig)


if __name__ == '__main__':
    run(clam=2.0)
