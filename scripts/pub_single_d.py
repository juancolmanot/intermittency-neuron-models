#!/usr/bin/env python3
"""
Individual publication figures for ALL d values.
One figure per panel per d: M, φ, ψ, l(u), branch fits.
Enlarged fonts, no d-value legends, publication-ready.
"""
import os, sys, numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from M_rpd_pdll_krause import (
    load_pts, load_laminar, fit_A, get_branches, krause_theory,
    BKPTS, DVALS, l_cont_func, pdll_pmf, alpha_from_slope, INV_OVERRIDES,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIG = os.path.join(BASE, 'figures')
CLAM = 2.0
C_NUM = 'blue'
C_THEO = 'red'

RC = {
    'font.size': 16,
    'axes.labelsize': 22,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'text.usetex': True,
    'font.family': 'serif',
}


def u0_from_l(l_val, ep, A_c, c_val):
    K = c_val**2 / (ep + A_c * c_val**2)
    denom = np.exp(2 * ep * l_val) - A_c * K
    u0sq = ep * K / np.clip(denom, 1e-30, None)
    return np.sqrt(np.clip(u0sq, 0, None))


def compute(D):
    eps_raw, pts = load_pts(D, CLAM)
    bkpts = BKPTS[D]
    inv_ov = INV_OVERRIDES.get(D)
    edges, branches = get_branches(pts, CLAM, bkpts, inv_override=inv_ov)
    x_theo, phi_theo, M_theo, ks, b_norm, alphas = krause_theory(branches, edges, CLAM)

    C_coeff, A_coeff = fit_A(D)
    eps_local = C_coeff - 1.0

    u_s = np.sort(pts)
    M_s = np.cumsum(u_s) / np.arange(1, len(u_s) + 1, dtype=float)

    nbins = 200
    counts, rpd_edges = np.histogram(pts, bins=nbins, range=(-CLAM, CLAM))
    rpd_c = 0.5 * (rpd_edges[:-1] + rpd_edges[1:])
    dw = rpd_edges[1] - rpd_edges[0]
    rpd_d = counts / (counts.sum() * dw)

    u_reinj, l_sim = load_laminar(D, CLAM)
    lmax = int(l_sim.max()) + 1
    l_arr, psi_num = pdll_pmf(l_sim, lmax)
    lmax_plot = int(np.percentile(l_sim, 99.5)) + 1

    _, phi_hr, _, _, _, _ = krause_theory(branches, edges, CLAM, ngrid=80000)
    x_hr = np.linspace(-CLAM, CLAM, 80000)
    phi_interp = lambda u, _xt=x_hr, _pt=phi_hr: np.interp(u, _xt, _pt, left=0.0, right=0.0)
    g_func = lambda u, _e=eps_local, _a=A_coeff: _e * u + _a * u**3
    nl_dense = min(lmax * 20, 100000)
    l_class = np.linspace(0.5, lmax, nl_dense)
    dl = l_class[1] - l_class[0]
    u0_class = u0_from_l(l_class, eps_local, A_coeff, CLAM)
    psi_class = np.array([
        (phi_interp(u) + phi_interp(-u)) * g_func(u) if u > 1e-10 else 0.0
        for u in u0_class
    ])
    psi_tot = psi_class.sum() * dl
    if psi_tot > 0:
        psi_class /= psi_tot

    psi_segments = []
    pos = psi_class > 0
    l_pos_t = l_class[pos]
    p_pos_t = psi_class[pos]
    if len(p_pos_t) > 1:
        log_pt = np.log10(p_pos_t)
        big_jump = np.abs(np.diff(log_pt)) > 0.3
        split_idx = np.where(big_jump)[0] + 1
        splits = np.split(np.arange(len(l_pos_t)), split_idx)
        for s in splits:
            if len(s) > 1:
                psi_segments.append((l_pos_t[s], p_pos_t[s]))
    elif len(p_pos_t) == 1:
        psi_segments.append((l_pos_t, p_pos_t))

    return dict(
        D=D, eps_local=eps_local, A_coeff=A_coeff,
        u_s=u_s, M_s=M_s, x_theo=x_theo, M_theo=M_theo,
        phi_theo=phi_theo, edges=edges, bkpts=bkpts,
        rpd_c=rpd_c, rpd_d=rpd_d,
        u_reinj=u_reinj, l_sim=l_sim,
        l_arr=l_arr, psi_num=psi_num, lmax_plot=lmax_plot,
        psi_segments=psi_segments, branches=branches,
    )


def tag(D):
    return D.replace('-', 'm').replace('.', 'p')


def make_M(r):
    matplotlib.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(7, 5))
    u_s, M_s = r['u_s'], r['M_s']
    step = max(1, len(u_s) // 5000)
    idx = np.arange(0, len(u_s), step)
    ax.plot(u_s[idx], M_s[idx], '.', color=C_NUM, ms=1.5, alpha=0.7, rasterized=True)
    ax.plot(r['x_theo'], r['M_theo'], '-', color=C_THEO, lw=1.5)
    for bk in r['bkpts']:
        ax.axvline(bk, color='gray', ls='--', lw=0.7, alpha=0.5)
    ax.set_xlim(-CLAM, CLAM)
    ax.set_xlabel(r'$\bar{u}$')
    ax.set_ylabel(r'$M(\bar{u})$')
    fig.tight_layout()
    out = os.path.join(FIG, f'pub_M_{tag(r["D"])}.pdf')
    fig.savefig(out)
    fig.savefig(out.replace('.pdf', '.png'), dpi=200)
    plt.close(fig)
    return out


def make_phi(r):
    matplotlib.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(r['rpd_c'], r['rpd_d'], '.', color=C_NUM, ms=3.0, alpha=0.7)
    ymax = r['rpd_d'].max() * 1.15
    phi_cut = ymax * 0.7
    x_theo, phi_theo, edges = r['x_theo'], r['phi_theo'], r['edges']
    for j in range(len(edges) - 1):
        lo_j, hi_j = edges[j], edges[j + 1]
        mask = (x_theo >= lo_j) & (x_theo <= hi_j) & \
               (phi_theo > 0) & (phi_theo <= phi_cut)
        ax.plot(x_theo[mask], phi_theo[mask], '-', color=C_THEO, lw=1.5)
    ax.set_xlim(-CLAM, CLAM)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(r'$\bar{u}$')
    ax.set_ylabel(r'$\varphi(\bar{u})$')
    fig.tight_layout()
    out = os.path.join(FIG, f'pub_phi_{tag(r["D"])}.pdf')
    fig.savefig(out)
    fig.savefig(out.replace('.pdf', '.png'), dpi=200)
    plt.close(fig)
    return out


def make_psi(r):
    matplotlib.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(7, 5))
    l_arr, psi_num = r['l_arr'], r['psi_num']
    lmax_plot = r['lmax_plot']
    clip = l_arr <= lmax_plot
    m_n = psi_num > 0
    ax.plot(l_arr[m_n & clip], psi_num[m_n & clip], '.', color='blue',
            ms=7.0, alpha=0.7, rasterized=True)
    for seg_l, seg_p in r['psi_segments']:
        m_s = seg_l <= lmax_plot
        if m_s.any():
            ax.plot(seg_l[m_s], seg_p[m_s], '-', color=C_THEO, lw=1.5)
    ax.set_xlim(0, lmax_plot)
    ax.set_xlabel(r'$l$')
    ax.set_ylabel(r'$\psi(l)$')
    fig.tight_layout()
    out = os.path.join(FIG, f'pub_psi_{tag(r["D"])}.pdf')
    fig.savefig(out)
    fig.savefig(out.replace('.pdf', '.png'), dpi=200)
    plt.close(fig)
    return out


def make_l_u(r):
    matplotlib.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(7, 5))
    u_reinj, l_sim = r['u_reinj'], r['l_sim']
    step_l = max(1, len(u_reinj) // 5000)
    idx_l = np.arange(0, len(u_reinj), step_l)
    ax.plot(u_reinj[idx_l], l_sim[idx_l], '.', color=C_NUM, ms=0.8,
            alpha=0.25, rasterized=True)
    u_grid = np.linspace(-CLAM + 0.01, CLAM - 0.01, 500)
    l_theo = l_cont_func(u_grid, r['eps_local'], r['A_coeff'], CLAM)
    ax.plot(u_grid, l_theo, '-', color=C_THEO, lw=1.5)
    ax.set_xlim(-CLAM, CLAM)
    l_99 = np.percentile(l_sim, 99.5)
    ax.set_ylim(0, l_99 * 1.1)
    ax.set_xlabel(r'$\bar{u}$')
    ax.set_ylabel(r'$l(\bar{u})$')
    fig.tight_layout()
    out = os.path.join(FIG, f'pub_lu_{tag(r["D"])}.pdf')
    fig.savefig(out)
    fig.savefig(out.replace('.pdf', '.png'), dpi=200)
    plt.close(fig)
    return out


def make_branch_fits(r):
    matplotlib.rcParams.update(RC)
    branches = r['branches']
    nbr = len(branches)
    fig, axes = plt.subplots(1, nbr, figsize=(6 * nbr, 5))
    if nbr == 1:
        axes = [axes]

    for j, br in enumerate(branches):
        ax = axes[j]
        if br is None:
            ax.set_visible(False)
            continue
        u_br = br['u']
        fit_lo_b = br.get('fit_lo', u_br.min())
        fit_hi_b = br.get('fit_hi', u_br.max())
        fm_b = (u_br >= fit_lo_b) & (u_br <= fit_hi_b)
        u_fit_pts = u_br[fm_b]
        if br['inverted']:
            refl = br['hi'] + (br['hi'] - u_fit_pts)
            M_clean_pts = np.cumsum(refl) / np.arange(1, len(refl)+1, dtype=float)
        else:
            M_clean_pts = np.cumsum(u_fit_pts) / np.arange(1, len(u_fit_pts)+1, dtype=float)
        step_b = max(1, len(u_fit_pts) // 5000)
        idx_b = np.arange(0, len(u_fit_pts), step_b)
        ax.plot(u_fit_pts[idx_b], M_clean_pts[idx_b], '.', color=C_NUM,
                ms=1.5, alpha=0.7, rasterized=True)
        m_clean = br.get('m_clean', br['m'])
        b_clean = br.get('b_int_clean', br['b_int'])
        u_fit = np.linspace(fit_lo_b, fit_hi_b, 200)
        M_fit = m_clean * u_fit + b_clean
        ax.plot(u_fit, M_fit, '-', color=C_THEO, lw=1.5)
        a_clean = alpha_from_slope(m_clean, br['inverted'])
        ax.text(0.05, 0.95,
                f'm={m_clean:.3f}\n' + r'$\alpha$' + f'={a_clean:.2f}',
                transform=ax.transAxes, fontsize=16, va='top', ha='left')
        ax.set_xlabel(r'$\bar{u}$')
        ax.set_ylabel(r'$M_{\mathrm{clean}}$')

    fig.tight_layout()
    out = os.path.join(FIG, f'pub_branches_{tag(r["D"])}.pdf')
    fig.savefig(out)
    fig.savefig(out.replace('.pdf', '.png'), dpi=200)
    plt.close(fig)
    return out


def main():
    for D in DVALS:
        print(f'\n=== d={D} ===', flush=True)
        r = compute(D)
        print(f'  {make_M(r)}')
        print(f'  {make_phi(r)}')
        print(f'  {make_psi(r)}')
        print(f'  {make_l_u(r)}')
        print(f'  {make_branch_fits(r)}')


if __name__ == '__main__':
    main()
