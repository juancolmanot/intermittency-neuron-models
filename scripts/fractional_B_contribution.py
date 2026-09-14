#!/usr/bin/env python3
"""
Fractional contribution of the B-term in F^2 polynomial fit.

Plots |B*u^2| / (|eps*u| + |B*u^2| + |A*u^3|) vs |u| for d = -12, -12.5, -13.
Black/white only, different linestyles, minimal legend.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Table 2 coefficients (exact, from two-pass fit procedure)
params = {
    -12.0: dict(A=0.0186, B=-0.0026, C=1.0716),
    -12.5: dict(A=0.0281, B=0.0266, C=1.2591),
    -13.0: dict(A=0.0505, B=0.0642, C=1.4920),
}

c = 2.0
u = np.linspace(0.01, c, 500)

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'font.family': 'serif',
    'font.size': 11,
})

fig, ax = plt.subplots(figsize=(5.5, 3.5))

styles = ['-', '--', '-.']
labels = []

for i, (d, p) in enumerate(sorted(params.items(), key=lambda x: x[0])):
    eps = p['C'] - 1.0
    B = abs(p['B'])
    A = abs(p['A'])

    term_eps = abs(eps) * u
    term_B = B * u**2
    term_A = A * u**3
    total = term_eps + term_B + term_A

    frac_B = term_B / total

    dlabel = f'{d:.1f}'.rstrip('0').rstrip('.')
    ax.plot(u, frac_B * 100, color='black', ls=styles[i], lw=1.8,
            label=r'$d = {}$'.format(dlabel))

ax.axhline(y=10, color='gray', ls=':', lw=0.8, alpha=0.6)

ax.set_xlabel(r'$|\bar{u}|$', fontsize=18)
ax.set_ylabel(r'$\dfrac{|B\bar{u}^2|}{|\varepsilon\bar{u}|+|B\bar{u}^2|+|A\bar{u}^3|}$ (\%)', fontsize=14)
ax.set_xlim(0, c)
ax.set_ylim(0, None)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(axis='both', labelsize=18)
ax.legend(frameon=False, fontsize=14, loc='upper left')

plt.tight_layout()

outdir = '/home/juan/Systems_Biology_Izhikevich/figures'
for fmt in ['png', 'pdf']:
    out = f'{outdir}/izhikevich_B_term_fraction.{fmt}'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved {out}')

plt.close(fig)
