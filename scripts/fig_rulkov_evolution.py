#!/usr/bin/env python3
"""Regenerate Figure 1: Rulkov map time evolution for four gamma values."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 14,
})

ALPHA = 4.8

def rulkov_map(gamma, N=500, x0=-1.8, transient=500):
    x = np.zeros(N + transient)
    x[0] = x0
    for i in range(N + transient - 1):
        xi = x[i]
        x[i+1] = ALPHA / (1.0 + xi*xi) + gamma
    return x[transient:]


gammas = [-2.93141, -2.931, -2.93, -2.927]
N = 500

fig, axes = plt.subplots(4, 1, figsize=(10, 5.5), sharex=True)
fig.subplots_adjust(hspace=0.15)

for i, (ax, g) in enumerate(zip(axes, gammas)):
    x = rulkov_map(g, N=N)
    n = np.arange(len(x))
    ax.plot(n, x, 'b-', lw=0.8)
    ax.set_ylabel(r'$x_n$', fontsize=20)
    ax.set_ylim(-3.5, 3.5)
    ax.tick_params(axis='both', labelsize=16)
    ax.locator_params(axis='y', nbins=3)
    label = rf'$\gamma = {g}$'
    ax.text(0.97, 0.92, label, transform=ax.transAxes, fontsize=18,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.9))

axes[-1].set_xlabel(r'$n$', fontsize=20)
axes[-1].tick_params(axis='x', labelsize=16)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'rulkov_map_evolution.png')
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved {out}")
