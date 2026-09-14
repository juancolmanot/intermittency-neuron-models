#!/usr/bin/env python3
"""
Regenerate ALL Rulkov c=0.1 analysis figures.

Produces (matching manuscript filenames):
  1. Return map           → figures/rulkov_map_c=0.1.png
  2. Local map zoom       → figures/rulkov_local_map_c=0.1.png
  3. M-function           → figures/rulkov_M_function_c=0.1.png
  4. RPD φ                → figures/rulkov_rpd_function_c=0.1.png
  5. PDLL ψ               → figures/rulkov_pdll_c=0.1.png

c=0.1 is the STANDARD single-branch reinjection case.
No piecewise treatment, no inset axes.
"""
import os
import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATADIR = os.path.join(BASE, 'code', 'datafiles')
FIGDIR = os.path.join(BASE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
})

# ── Constants ─────────────────────────────────────────────────────────
ALPHA_MAP = 4.8
GAMMA_C = -2.931400258306671
FP = -1.765458470442114
A2 = -ALPHA_MAP * (1 - 3*FP**2) / (1 + FP**2)**3
c = 0.1

print(f"x* = {FP:.15f}")
print(f"gamma_c = {GAMMA_C:.15f}")
print(f"a2 = {A2:.8f}")


def F(x_abs, gamma):
    return ALPHA_MAP / (1.0 + x_abs * x_abs) + gamma


# ── 1. Load or generate reinjection data ─────────────────────────────
EPS_REINJ = 3e-4
gamma_reinj = GAMMA_C + EPS_REINJ

reinj_file = os.path.join(DATADIR, 'rulkov_reinjection_c=0.10.dat')
if os.path.isfile(reinj_file):
    print(f"\n{'='*60}")
    print(f"1. Loading reinjection data from {reinj_file}")
    reinj = np.loadtxt(reinj_file, comments='#')
    print(f"  Loaded {len(reinj)} reinjections")
else:
    print(f"\n{'='*60}")
    TARGET = 200000
    print(f"1. Generating reinjection data (eps={EPS_REINJ}, target={TARGET})")

    np.random.seed(42)
    x = 0.5
    for _ in range(500000):
        x = F(x, gamma_reinj)

    reinj = []
    lam_lo, lam_hi = FP - c, FP + c
    in_lam = (lam_lo <= x <= lam_hi)
    step = 0
    while len(reinj) < TARGET:
        x = F(x, gamma_reinj)
        now_in = (lam_lo <= x <= lam_hi)
        if now_in and not in_lam:
            reinj.append(x - FP)
        in_lam = now_in
        step += 1
        if step > 500_000_000:
            print(f"  WARNING: reached 500M steps with {len(reinj)} reinjections")
            break

    reinj = np.array(reinj)
    header = f"c={c:.6f} eps={EPS_REINJ:.8e} gamma={gamma_reinj:.10f} fp={FP:.15f}"
    np.savetxt(reinj_file, reinj, header=header, fmt='%.15e')
    print(f"  Generated {len(reinj)} reinjections → {reinj_file}")

print(f"  Range: [{reinj.min():.8f}, {reinj.max():.8f}]")

# ── 2. M-function analysis (SINGLE branch for c=0.1) ────────────────
print(f"\n{'='*60}")
print("2. M-function analysis (single branch)")

x_in = reinj[(reinj >= -c) & (reinj <= c)]
x_sorted = np.sort(x_in)
N = len(x_sorted)

M_vals = np.cumsum(x_sorted) / np.arange(1, N + 1)

x_i = x_sorted[0]
m_fit, b_fit, r_fit, _, _ = stats.linregress(x_sorted, M_vals)
alpha_rpd = (2*m_fit - 1) / (1 - m_fit)
print(f"  x_i = {x_i:.6f}")
print(f"  m = {m_fit:.4f}, α = {alpha_rpd:.4f}, R² = {r_fit**2:.6f}")

# ── 3. RPD φ (single power law) ─────────────────────────────────────
print(f"\n{'='*60}")
print("3. RPD computation (single branch)")

nbins = 200
counts, edges = np.histogram(x_in, bins=nbins, range=(-c, c))
centers = 0.5 * (edges[:-1] + edges[1:])
dw = edges[1] - edges[0]
phi_num = counts / (counts.sum() * dw)

I_raw = (c - x_i)**(alpha_rpd + 1) / (alpha_rpd + 1)
b_phi = 1.0 / I_raw
norm_check = b_phi * I_raw
print(f"  b = {b_phi:.4f}")
print(f"  normalization check: {norm_check:.6f}")


def phi_theo(x_val):
    if x_val < x_i or x_val > c:
        return 0.0
    dx = x_val - x_i
    if dx <= 0:
        dx = 1e-15
    return b_phi * dx**alpha_rpd


x_theo = np.linspace(x_i + 1e-8, c, 1000)
phi_theo_arr = np.array([phi_theo(xv) for xv in x_theo])

# ── 4. PDLL ψ ────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("4. PDLL computation")

lam_lo, lam_hi = FP - c, FP + c

np.random.seed(42)
x = 0.5
for _ in range(500000):
    x = F(x, gamma_reinj)

lam_lengths = []
current_len = 0
in_lam = (lam_lo <= x <= lam_hi)

for _ in range(80_000_000):
    x = F(x, gamma_reinj)
    now_in = (lam_lo <= x <= lam_hi)
    if now_in:
        current_len += 1
    else:
        if in_lam and current_len > 0:
            lam_lengths.append(current_len)
        current_len = 0
    in_lam = now_in

lam_lengths = np.array(lam_lengths)
print(f"  Collected {len(lam_lengths)} laminar episodes")
print(f"  Mean length: {lam_lengths.mean():.1f}")
print(f"  Max length: {lam_lengths.max()}")

l_max_plot = min(int(np.percentile(lam_lengths, 99.5)), lam_lengths.max())
nbins_psi = min(100, l_max_plot)
counts_psi, edges_psi = np.histogram(lam_lengths, bins=nbins_psi,
                                      range=(1, l_max_plot), density=True)
centers_psi = 0.5 * (edges_psi[:-1] + edges_psi[1:])


def l_type1(x_entry, c_val, a2_val, eps_val):
    if a2_val * eps_val <= 0:
        return 0.0
    sqae = np.sqrt(a2_val * eps_val)
    ratio = np.sqrt(a2_val / eps_val)
    return (np.arctan(c_val * ratio) - np.arctan(x_entry * ratio)) / sqae


x_entry_grid = np.linspace(x_i + 1e-8, c - 1e-8, 50000)
l_of_x = np.array([l_type1(xe, c, A2, EPS_REINJ) for xe in x_entry_grid])

l_grid = np.linspace(1, l_max_plot, 500)
psi_theo = np.zeros_like(l_grid)

valid = l_of_x > 0
if valid.sum() > 10:
    l_interp = interp1d(l_of_x[valid][::-1], x_entry_grid[valid][::-1],
                        bounds_error=False, fill_value=np.nan)
    x_of_l = l_interp(l_grid)

    for i, (l_val, x_val) in enumerate(zip(l_grid, x_of_l)):
        if np.isnan(x_val):
            continue
        phi_val = phi_theo(x_val)
        psi_theo[i] = phi_val * (EPS_REINJ + A2 * x_val**2)

# ── 5. Generate return map data ──────────────────────────────────────
print(f"\n{'='*60}")
print("5. Generating return map data for figures")

np.random.seed(42)
x = 0.5
for _ in range(500000):
    x = F(x, gamma_reinj)

map_x = []
map_y = []
for _ in range(10_000_000):
    x_prev = x
    x = F(x, gamma_reinj)
    map_x.append(x_prev)
    map_y.append(x)

map_x = np.array(map_x)
map_y = np.array(map_y)

# ======================================================================
# 6. PUBLICATION FIGURES
# ======================================================================
print(f"\n{'='*60}")
print("6. Generating publication figures")

TICK_SIZE = 30
LABEL_SIZE = 39
MS = 40
LW_THEO = 3.0


def set_ticks(ax, n=4):
    ax.tick_params(axis='both', labelsize=TICK_SIZE, which='both',
                   direction='in', top=True, right=True)
    ax.locator_params(axis='x', nbins=n)
    ax.locator_params(axis='y', nbins=n)


# ── 6a. Return map (global) — 10x more points ───────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ss = max(1, len(map_x) // 200000)
ax.scatter(map_x[::ss], map_y[::ss], s=3, c='blue', alpha=0.2,
           edgecolors='none', rasterized=True, zorder=2)
xlim = [map_x.min(), map_x.max()]
ax.plot(xlim, xlim, 'k--', lw=0.8, alpha=0.5)
ax.set_xlabel(r'$x_n$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$x_{n+1}$', fontsize=LABEL_SIZE)
set_ticks(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'rulkov_map_c=0.1.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGDIR, 'rulkov_map_c=0.1.pdf'), bbox_inches='tight')
plt.close(fig)
print("  Saved rulkov_map_c=0.1")

# ── 6b. Local map zoom (centered variable) ──────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
cx = map_x - FP
cy = map_y - FP
local_mask = (np.abs(cx) < c * 1.5) & (np.abs(cy) < c * 1.5)
cx_l = cx[local_mask]
cy_l = cy[local_mask]
ss_l = max(1, len(cx_l) // 50000)
ax.scatter(cx_l[::ss_l], cy_l[::ss_l], s=MS, c='blue', alpha=0.5,
           edgecolors='none', zorder=2)

x_anal = np.linspace(-c * 1.2, c * 1.2, 500)
y_anal = A2 * x_anal**2 + x_anal + EPS_REINJ
ax.plot(x_anal, y_anal, 'k-', lw=LW_THEO, zorder=3)
ax.plot([-c*1.5, c*1.5], [-c*1.5, c*1.5], 'k--', lw=0.8, alpha=0.5)
ax.axvline(-c, color='gray', ls=':', lw=0.8, alpha=0.5)
ax.axvline(c, color='gray', ls=':', lw=0.8, alpha=0.5)
ax.set_xlabel(r'$x_n - x^*$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$x_{n+1} - x^*$', fontsize=LABEL_SIZE)
set_ticks(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'rulkov_local_map_c=0.1.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGDIR, 'rulkov_local_map_c=0.1.pdf'), bbox_inches='tight')
plt.close(fig)
print("  Saved rulkov_local_map_c=0.1")

# ── 6c. M-function (single branch, single fit line) ─────────────────
fig, ax = plt.subplots(figsize=(8, 6))
subsample = max(1, len(x_sorted) // 5000)
ax.scatter(x_sorted[::subsample], M_vals[::subsample], s=MS, c='blue',
           marker='o', edgecolors='none', alpha=0.7)

M_fit_line = m_fit * (x_sorted - x_i) + x_i
ax.plot(x_sorted, M_fit_line, '-', color='red', lw=LW_THEO, zorder=5,
        label=rf'$m = {m_fit:.3f}$')

ax.set_xlabel(r'$x$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$M(x)$', fontsize=LABEL_SIZE)
ax.legend(fontsize=TICK_SIZE, loc='upper left')
set_ticks(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'rulkov_M_function_c=0.1.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGDIR, 'rulkov_M_function_c=0.1.pdf'), bbox_inches='tight')
plt.close(fig)
print("  Saved rulkov_M_function_c=0.1")

# ── 6d. RPD φ (single curve, no inset) ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
pos = phi_num > 0
ax.scatter(centers[pos], phi_num[pos], s=MS, c='blue',
           marker='o', edgecolors='none', alpha=0.7, zorder=2)
ax.plot(x_theo, phi_theo_arr, 'r-', lw=LW_THEO, zorder=3)
ax.set_xlabel(r'$x$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$\phi(x)$', fontsize=LABEL_SIZE)
ax.set_ylim(0, phi_num[pos].max() * 1.15)
set_ticks(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'rulkov_rpd_function_c=0.1.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGDIR, 'rulkov_rpd_function_c=0.1.pdf'), bbox_inches='tight')
plt.close(fig)
print("  Saved rulkov_rpd_function_c=0.1")

# ── 6e. PDLL ψ (single curve, no inset) ─────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(centers_psi, counts_psi, s=MS, c='blue',
           marker='o', edgecolors='none', alpha=0.7, zorder=2)
valid_psi = psi_theo > 0
ax.plot(l_grid[valid_psi], psi_theo[valid_psi], 'r-', lw=LW_THEO, zorder=3)
ax.set_xlabel(r'$l$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$\psi(l)$', fontsize=LABEL_SIZE)
set_ticks(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'rulkov_pdll_c=0.1.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGDIR, 'rulkov_pdll_c=0.1.pdf'), bbox_inches='tight')
plt.close(fig)
print("  Saved rulkov_pdll_c=0.1")

print(f"\n{'='*60}")
print("DONE. Key results:")
print(f"  x* = {FP:.15f}")
print(f"  c = {c}")
print(f"  a2 = {A2:.8f}")
print(f"  x_i = {x_i:.6f}")
print(f"  M slope: m = {m_fit:.4f}")
print(f"  RPD exponent: α = {alpha_rpd:.4f}")
print(f"  <l> at eps={EPS_REINJ}: {lam_lengths.mean():.1f} (sim)")
print(f"{'='*60}")
