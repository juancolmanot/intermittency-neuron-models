#!/usr/bin/env python3
"""
Publication figures for d=-11.90: 6 individual plots matching comprehensive figure data.
Uses identical pipeline as pub_comprehensive_all_d.py.
"""
import os, sys, numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from M_rpd_pdll_krause import (
    load_pts, load_laminar, fit_A, get_branches, krause_theory,
    BKPTS, INV_OVERRIDES, PAD_OVERRIDES, l_cont_func, pdll_pmf, alpha_from_slope,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.size': 28,
    'axes.labelsize': 38,
    'xtick.labelsize': 26,
    'ytick.labelsize': 26,
    'legend.fontsize': 26,
    'text.usetex': True,
    'font.family': 'serif',
})

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIG = os.path.join(BASE, 'figures')

D = '-11.90'
CLAM = 2.0
C_NUM = 'blue'
C_THEO = 'red'


def u0_from_l(l_val, ep, A_c, c_val):
    K = c_val**2 / (ep + A_c * c_val**2)
    denom = np.exp(2 * ep * l_val) - A_c * K
    u0sq = ep * K / np.clip(denom, 1e-30, None)
    return np.sqrt(np.clip(u0sq, 0, None))


def main():
    eps, pts = load_pts(D, CLAM)
    bkpts = BKPTS[D]
    inv_ov = INV_OVERRIDES.get(D)
    pad_ov = PAD_OVERRIDES.get(D)
    edges, branches = get_branches(pts, CLAM, bkpts,
                                   inv_override=inv_ov, pad_override=pad_ov)
    x_theo, phi_theo, M_theo, ks, b_norm, alphas = krause_theory(
        branches, edges, CLAM)

    C_coeff, A_coeff = fit_A(D)
    eps_local = C_coeff - 1.0

    # Numerical M
    u_s = np.sort(pts)
    M_s = np.cumsum(u_s) / np.arange(1, len(u_s) + 1, dtype=float)

    # Numerical RPD
    nbins = 200
    counts, rpd_edges = np.histogram(pts, bins=nbins, range=(-CLAM, CLAM))
    rpd_c = 0.5 * (rpd_edges[:-1] + rpd_edges[1:])
    dw = rpd_edges[1] - rpd_edges[0]
    rpd_d = counts / (counts.sum() * dw)

    # Laminar data
    u_reinj, l_sim = load_laminar(D, CLAM)
    lmax = int(l_sim.max()) + 1
    l_arr, psi_num = pdll_pmf(l_sim, lmax)
    lmax_plot = int(np.percentile(l_sim, 99.5)) + 1

    # Classical ψ — high-res φ pushforward (same as comprehensive figure)
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

    # Split theoretical ψ into segments at discontinuities
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

    # ── Fig 1: Global M ──────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    step = max(1, len(u_s) // 5000)
    idx = np.arange(0, len(u_s), step)
    ax1.plot(u_s[idx], M_s[idx], '.', color=C_NUM, ms=2.5, alpha=1.0,
             rasterized=True)
    ax1.plot(x_theo, M_theo, '-', color=C_THEO, lw=1.8)
    for bk in bkpts:
        ax1.axvline(bk, color='gray', ls='--', lw=0.6, alpha=0.4)
    ax1.set_xlabel(r"$\bar{u}$")
    ax1.set_ylabel(r"$M(\bar{u})$")
    ax1.set_xlim(-CLAM, CLAM)
    fig1.tight_layout()
    out1 = os.path.join(FIG, 'pub_izh_M_d1190.png')
    fig1.savefig(out1, dpi=200)
    fig1.savefig(out1.replace('.png', '.pdf'))
    print(f'Wrote {out1}')
    plt.close(fig1)

    # ── Fig 2: RPD φ ─────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(rpd_c, rpd_d, '.', color=C_NUM, ms=5, alpha=1.0)
    ymax = rpd_d.max() * 1.15
    phi_cut = ymax * 0.7
    for j in range(len(edges) - 1):
        lo_j, hi_j = edges[j], edges[j + 1]
        mask = (x_theo >= lo_j) & (x_theo <= hi_j) & \
               (phi_theo > 0) & (phi_theo <= phi_cut)
        ax2.plot(x_theo[mask], phi_theo[mask], '-', color=C_THEO, lw=1.8)
    ax2.set_xlabel(r"$\bar{u}$")
    ax2.set_ylabel(r"$\phi(\bar{u})$")
    ax2.set_xlim(-CLAM, CLAM)
    ax2.set_ylim(0, ymax)
    fig2.tight_layout()
    ax_ins = fig2.add_axes([0.50, 0.45, 0.40, 0.28])
    ins_mask = (rpd_c >= -0.5) & (rpd_c <= 2.0) & (rpd_d <= 0.05)
    ax_ins.plot(rpd_c[ins_mask], rpd_d[ins_mask], '.', color=C_NUM, ms=3, alpha=0.8)
    ins_t = (x_theo >= -0.5) & (x_theo <= 2.0) & (phi_theo > 0) & (phi_theo <= 0.05)
    ax_ins.plot(x_theo[ins_t], phi_theo[ins_t], '-', color=C_THEO, lw=1.5)
    ax_ins.set_xlim(-0.5, 2.05)
    ax_ins.set_ylim(0, 0.05)
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])
    ax_ins.patch.set_facecolor('white')
    ax_ins.patch.set_alpha(1.0)
    out2 = os.path.join(FIG, 'pub_izh_phi_d1190.png')
    fig2.savefig(out2, dpi=200)
    fig2.savefig(out2.replace('.png', '.pdf'))
    print(f'Wrote {out2}')
    plt.close(fig2)

    # ── Fig 3: PDLL ψ (linear y, inset for tail) ─────────────────────
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    clip = l_arr <= lmax_plot
    m_n = psi_num > 0
    ax3.plot(l_arr[m_n & clip], psi_num[m_n & clip], 'o', color=C_NUM,
             ms=5, alpha=1.0, rasterized=True, zorder=4)
    for k, (sl, sp) in enumerate(psi_segments):
        m_s = sl <= lmax_plot
        if m_s.any():
            ax3.plot(sl[m_s], sp[m_s], '-', color=C_THEO, lw=1.8,
                     alpha=0.9, zorder=2)
    ax3.set_xlabel(r"$l$")
    ax3.set_ylabel(r"$\psi(l)$")
    ax3.set_xlim(0, lmax_plot)
    fig3.tight_layout()
    out3 = os.path.join(FIG, 'pub_izh_pdll_d1190.png')
    fig3.savefig(out3, dpi=200)
    fig3.savefig(out3.replace('.png', '.pdf'))
    print(f'Wrote {out3}')
    plt.close(fig3)

    # ── Fig 4: l(ū) scatter ──────────────────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    step_l = max(1, len(u_reinj) // 5000)
    idx_l = np.arange(0, len(u_reinj), step_l)
    ax4.plot(u_reinj[idx_l], l_sim[idx_l], '.', color=C_NUM, ms=1.5,
             alpha=0.3, rasterized=True)
    u_grid = np.linspace(-CLAM + 0.01, CLAM - 0.01, 500)
    l_theo = l_cont_func(u_grid, eps_local, A_coeff, CLAM)
    ax4.plot(u_grid, l_theo, '-', color=C_THEO, lw=1.8)
    ax4.set_xlim(-CLAM, CLAM)
    l_99 = np.percentile(l_sim, 99.5)
    ax4.set_ylim(0, l_99 * 1.1)
    ax4.set_xlabel(r"$\bar{u}$")
    ax4.set_ylabel(r"$l(\bar{u})$")
    fig4.tight_layout()
    out4 = os.path.join(FIG, 'pub_izh_l_d1190.png')
    fig4.savefig(out4, dpi=200)
    fig4.savefig(out4.replace('.png', '.pdf'))
    print(f'Wrote {out4}')
    plt.close(fig4)

    # ── Fig 5 & 6: Branch M fits (individual) ────────────────────────
    labels = ['(a)', '(b)', '(c)', '(d)']
    nbr = len(branches)
    # Also combined figure for manuscript compatibility
    fig_br, axes_br = plt.subplots(1, nbr, figsize=(5 * nbr, 4))
    if nbr == 1:
        axes_br = [axes_br]

    for j, br in enumerate(branches):
        if br is None:
            continue
        u_br, M_br = br['u'], br['M']
        fit_lo_b = br.get('fit_lo', u_br.min())
        fit_hi_b = br.get('fit_hi', u_br.max())
        fm_b = (u_br >= fit_lo_b) & (u_br <= fit_hi_b)
        u_fit_pts = u_br[fm_b]
        if br['inverted']:
            refl = br['hi'] + (br['hi'] - u_fit_pts)
            M_clean_pts = np.cumsum(refl) / np.arange(1, len(refl)+1, dtype=float)
        else:
            M_clean_pts = np.cumsum(u_fit_pts) / np.arange(1, len(u_fit_pts)+1, dtype=float)

        m_clean = br.get('m_clean', br['m'])
        b_clean = br.get('b_int_clean', br['b_int'])
        a_clean = alpha_from_slope(m_clean, br['inverted'])
        u_fit = np.linspace(fit_lo_b, fit_hi_b, 200)
        M_fit = m_clean * u_fit + b_clean

        # Individual figure
        figj, axj = plt.subplots(figsize=(6, 4.5))
        step_b = max(1, len(u_fit_pts) // 5000)
        idx_b = np.arange(0, len(u_fit_pts), step_b)
        axj.plot(u_fit_pts[idx_b], M_clean_pts[idx_b], '.', color=C_NUM,
                 ms=2.5, alpha=1.0, rasterized=True)
        axj.plot(u_fit, M_fit, '-', color=C_THEO, lw=2.0)
        axj.set_xlabel(r"$\bar{u}$")
        axj.set_ylabel(rf"$M_{j+1}(\bar{{u}})$")
        axj.text(0.03, 0.97, rf'$\alpha_{j+1} = {a_clean:.3f}$',
                 transform=axj.transAxes, fontsize=30, va='top')
        figj.tight_layout()
        outj = os.path.join(FIG, f'pub_izh_M{j+1}_d1190.png')
        figj.savefig(outj, dpi=200)
        figj.savefig(outj.replace('.png', '.pdf'))
        print(f'Wrote {outj}')
        plt.close(figj)

        # Combined figure panel
        ax = axes_br[j]
        ax.plot(u_fit_pts[idx_b], M_clean_pts[idx_b], '.', color=C_NUM,
                ms=2.5, alpha=1.0, rasterized=True)
        ax.plot(u_fit, M_fit, '-', color=C_THEO, lw=2.0)
        ax.set_xlabel(r"$\bar{u}$")
        ax.set_ylabel(rf"$M_{j+1}$")
        ax.text(0.03, 0.97, labels[j], transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')

    fig_br.tight_layout()
    out_br = os.path.join(FIG, 'pub_izh_branches_d1190.png')
    fig_br.savefig(out_br, dpi=200)
    fig_br.savefig(out_br.replace('.png', '.pdf'))
    print(f'Wrote {out_br}')
    plt.close(fig_br)


if __name__ == '__main__':
    main()
