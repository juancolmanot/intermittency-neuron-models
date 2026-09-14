#!/usr/bin/env python3
"""
rulkov_bifurcation_pub.py — Publication-quality Floquet multiplier figure.

Shows type-I intermittency mechanism in Rulkov fast map:
  (a) Saddle-node fold of fixed points
  (b) Real Floquet multiplier → +1 at fold
  (c) Tangent channel after bifurcation

Style matches manuscript figures: all spines, LaTeX text, blue/red/black.
"""
import os
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..'))
FIG  = os.path.join(BASE, 'figures')
os.makedirs(FIG, exist_ok=True)

ALPHA = 4.8

# Manuscript palette: blue data, red theory, black reference
C_BLUE  = '#0000FF'
C_RED   = '#CC0000'
C_BLACK = '#000000'
C_GRAY  = '#666666'


def F(x, g):     return ALPHA / (1 + x**2) + g
def Fp(x):       return -2 * ALPHA * x / (1 + x**2)**2
def Fpp(x):      return -2 * ALPHA * (1 - 3 * x**2) / (1 + x**2)**3


def solve_tangency():
    g = lambda x: Fp(x) - 1.0
    xstar = brentq(g, -3.0, -0.5, xtol=1e-14, rtol=1e-15)
    gamma_c = xstar - ALPHA / (1 + xstar**2)
    return xstar, gamma_c


def real_fixed_points(g):
    roots = np.roots([1.0, -g, 1.0, -(g + ALPHA)])
    return np.sort(roots[np.abs(roots.imag) < 1e-9].real)


def setup_style():
    mpl.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'lines.linewidth': 1.5,
    })


def main():
    setup_style()
    xstar, gamma_c = solve_tangency()
    a2 = 0.5 * Fpp(xstar)

    print(f'gamma_c = {gamma_c:.12f}')
    print(f'x*      = {xstar:.12f}')
    print(f"F'(x*)  = {Fp(xstar):.12f}")
    print(f'a2      = {a2:.6f}')

    # Compute branches
    gs = np.linspace(gamma_c - 0.25, gamma_c + 0.15, 2000)
    stable_g, stable_x, saddle_g, saddle_x = [], [], [], []
    for g in gs:
        fps = real_fixed_points(g)
        for x in fps:
            if -3.5 < x < 0.5:
                if abs(Fp(x)) < 1:
                    stable_g.append(g); stable_x.append(x)
                else:
                    saddle_g.append(g); saddle_x.append(x)

    stable_g = np.array(stable_g); stable_x = np.array(stable_x)
    saddle_g = np.array(saddle_g); saddle_x = np.array(saddle_x)

    si = np.argsort(stable_g); stable_g = stable_g[si]; stable_x = stable_x[si]
    ui = np.argsort(saddle_g); saddle_g = saddle_g[ui]; saddle_x = saddle_x[ui]

    mult_stable = np.array([Fp(x) for x in stable_x])

    # --- Figure: double-column width ---
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    ax_a, ax_b, ax_c = axes

    # Panel (a): Saddle-node fold
    ax_a.plot(stable_g, stable_x, '-', color=C_BLUE, lw=1.8, label=r'Stable FP')
    ax_a.plot(saddle_g, saddle_x, '-', color=C_RED, lw=1.8, label=r'Unstable FP')
    ax_a.plot(gamma_c, xstar, 'ks', ms=6, zorder=5,
              label=r'Fold $\gamma_c$')
    ax_a.axvline(gamma_c, color=C_BLACK, ls='--', lw=0.5, alpha=0.4)
    ax_a.set_xlabel(r'$\gamma$')
    ax_a.set_ylabel(r'$x^*$')
    ax_a.legend(frameon=False, loc='center left', fontsize=7.5,
                handlelength=1.5)

    ax_a.annotate(r'FPs collide',
                  xy=(gamma_c, xstar),
                  xytext=(gamma_c - 0.17, xstar + 0.5),
                  fontsize=8,
                  arrowprops=dict(arrowstyle='->', lw=0.7))

    # Panel (b): Multiplier
    ax_b.plot(stable_g, mult_stable, '-', color=C_BLUE, lw=1.8)
    ax_b.axhline(1.0, color=C_BLACK, ls='--', lw=0.7)
    ax_b.axvline(gamma_c, color=C_BLACK, ls='--', lw=0.5, alpha=0.4)
    ax_b.plot(gamma_c, 1.0, 'ks', ms=6, zorder=5)
    ax_b.set_xlabel(r'$\gamma$')
    ax_b.set_ylabel(r"$F'(x^*)$")
    ax_b.set_ylim(0.42, 1.08)

    ax_b.annotate(r"$F'(x^*) = +1$",
                  xy=(gamma_c, 1.0),
                  xytext=(gamma_c - 0.17, 0.72),
                  fontsize=9,
                  arrowprops=dict(arrowstyle='->', lw=0.7))

    # Panel (c): Return map tangent channel
    eps = 5e-3
    xs = np.linspace(xstar - 0.6, xstar + 0.6, 1000)
    ax_c.plot(xs, F(xs, gamma_c + eps), '-', color=C_BLUE, lw=1.8,
              label=r'$F(x)$')
    ax_c.plot(xs, xs, '--', color=C_BLACK, lw=1.0, label=r'$y = x$')
    ax_c.set_xlabel(r'$x_n$')
    ax_c.set_ylabel(r'$x_{n+1}$')
    ax_c.legend(frameon=False, loc='upper left', fontsize=8)

    # Channel gap annotation
    x_ch = xstar + 0.05
    y_f = F(x_ch, gamma_c + eps)
    ax_c.annotate('', xy=(x_ch, y_f), xytext=(x_ch, x_ch),
                  arrowprops=dict(arrowstyle='<->', color=C_RED, lw=1.0))
    ax_c.text(x_ch + 0.12, 0.5*(x_ch + y_f),
              r'\textit{channel}', fontsize=8, color=C_RED, va='center')

    # Panel labels (a), (b), (c) — manuscript style
    for i, ax in enumerate(axes):
        label = chr(ord('a') + i)
        ax.text(-0.17, 1.06, r'\textbf{(' + label + r')}',
                transform=ax.transAxes, fontsize=11, va='top')

    fig.tight_layout(w_pad=1.5)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'rulkov_floquet_pub.{ext}'),
                    dpi=300, bbox_inches='tight')
    print(f'Wrote figures/rulkov_floquet_pub.png/pdf')


if __name__ == '__main__':
    main()
