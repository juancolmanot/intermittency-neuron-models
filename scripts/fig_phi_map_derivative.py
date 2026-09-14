#!/usr/bin/env python3
"""
Two-panel figure: RPD accumulation peak linked to low-derivative region of F^2.
Left panel: phi(u_bar) rotated 90° clockwise (density on X, u_bar on Y).
Right panel: F^2 outer branch zoomed to low-derivative region, shared Y axis.
Horizontal red line connects the extremum output to the RPD peak.
"""
import os, sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIG = os.path.join(BASE, 'figures')
SCALING = os.path.join(BASE, 'cpp', 'out', 'scaling')

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 18,
    'axes.labelsize': 26,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
})


def make_figure():
    # --- Load RPD data (d=-11.90, clam=2.0) ---
    ep = np.loadtxt(os.path.join(SCALING, 'd-11.90_ep_clam2.00.dat'), comments='#')
    u0 = ep[:, 0]
    clam = 2.0
    pts = u0[(u0 > -clam) & (u0 < clam)]

    nbins = 200
    counts, edges = np.histogram(pts, bins=nbins, range=(-clam, clam))
    centers = 0.5 * (edges[:-1] + edges[1:])
    dw = edges[1] - edges[0]
    rpd = counts / (counts.sum() * dw)

    peak_idx = np.argmax(rpd)
    peak_u = centers[peak_idx]
    print(f"RPD peak at u_bar = {peak_u:.4f}")

    # --- Load F^2 map data (d=-12, full multi-IC) ---
    mdata = np.loadtxt(os.path.join(BASE, 'code', 'datafiles', 'poincare_fig19_d=-12.dat'))
    ui = mdata[:, 0]
    ui2 = mdata[:, 2]
    FP = -98.1265
    ui_s = ui - FP
    ui2_s = ui2 - FP

    idx = np.argsort(ui_s)
    xs = ui_s[idx]
    ys = ui2_s[idx]

    # Full branch including laminar region and outer reinjection
    zoom_mask = (xs > -clam) & (xs < 11.0)
    x_zoom = xs[zoom_mask]
    y_zoom = ys[zoom_mask]

    # Compute derivative only in outer branch (x > clam) to find extremum
    outer_mask = x_zoom > clam
    x_outer = x_zoom[outer_mask]
    y_outer = y_zoom[outer_mask]
    deriv = np.gradient(y_outer, x_outer)
    min_deriv_idx = np.argmin(np.abs(deriv))
    extremum_x = x_outer[min_deriv_idx]
    extremum_y = y_outer[min_deriv_idx]
    print(f"Extremum at x={extremum_x:.3f}, y={extremum_y:.3f}, |deriv|={np.abs(deriv[min_deriv_idx]):.5f}")

    # --- Figure ---
    fig, (ax_rpd, ax_map) = plt.subplots(1, 2, figsize=(12, 6),
                                          sharey=True,
                                          gridspec_kw={'width_ratios': [1, 1.3], 'wspace': 0.05})

    # Left panel: RPD rotated — density on X pointing LEFT (toward map panel)
    pos = rpd > 0
    ax_rpd.plot(rpd[pos], centers[pos], '.', color='blue', ms=5, alpha=0.7, rasterized=True)
    ax_rpd.set_xlabel(r'$\phi(\bar{u})$')
    ax_rpd.set_ylabel(r'$\bar{u}$')
    ax_rpd.set_ylim(peak_u - 2.0, peak_u + 2.0)
    ax_rpd.set_xlim(rpd[pos].max() * 1.1, 0)
    ax_rpd.text(0.95, 0.95, r'(a)', transform=ax_rpd.transAxes,
                fontsize=22, va='top', ha='right', fontweight='bold')

    # Right panel: F^2 branch — wider view around turning point
    ylim_lo, ylim_hi = peak_u - 2.0, peak_u + 2.0
    vis = (y_zoom > ylim_lo) & (y_zoom < ylim_hi)
    step = max(1, vis.sum() // 4000)
    ax_map.plot(x_zoom[vis][::step], y_zoom[vis][::step], '.', color='blue',
                ms=6, alpha=0.7, rasterized=True)
    ax_map.set_xlabel(r'$\bar{u}_i$')
    ax_map.set_ylabel(r'$\bar{u}_{i+2}$')
    ax_map.yaxis.set_label_position('right')
    ax_map.yaxis.tick_right()
    ax_map.text(0.05, 0.95, r'(b)', transform=ax_map.transAxes,
                fontsize=22, va='top', fontweight='bold')

    # Horizontal red line across both panels
    for ax in (ax_rpd, ax_map):
        ax.axhline(extremum_y, color='red', lw=2.0, ls='--', alpha=0.8, zorder=10)

    fig.tight_layout()
    out = os.path.join(FIG, 'phi_map_derivative_link.pdf')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")
    return out


if __name__ == '__main__':
    make_figure()
