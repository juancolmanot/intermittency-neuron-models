#!/usr/bin/env python3
"""
Regenerate ALL Rulkov c=0.01 analysis with correct x*.

Produces:
  1. Reinjection data  → datafiles/rulkov_reinjection_c=0.01.dat
  2. M-function        → figures/M_f_c=0.01.png (+ per-branch)
  3. RPD φ             → figures/rpd_f_c=0.01.png
  4. PDLL ψ            → figures/pdll_f_c=0.01.png
  5. Scaling            → figures/rulkov_scaling_v2_c=0.01.png
  6. Summary panel      → figures/rulkov_c001_summary.png
"""
import os
import numpy as np
from scipy import stats
from scipy.integrate import quad
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
    "font.size": 11,
})

# ── Correct constants ──────────────────────────────────────────────────
ALPHA_MAP = 4.8
GAMMA_C = -2.931400258306671
FP = -1.765458470442114
A2 = -ALPHA_MAP * (1 - 3*FP**2) / (1 + FP**2)**3
c = 0.01

print(f"x* = {FP:.15f}")
print(f"gamma_c = {GAMMA_C:.15f}")
print(f"a2 = {A2:.8f}")


# ── 1. Reinjection data ───────────────────────────────────────────────
def F(x_abs, gamma):
    return ALPHA_MAP / (1.0 + x_abs * x_abs) + gamma

EPS_REINJ = 3e-4
gamma_reinj = GAMMA_C + EPS_REINJ
TARGET = 200000

print(f"\n{'='*60}")
print(f"1. Generating reinjection data (eps={EPS_REINJ}, target={TARGET})")
print(f"   gamma = {gamma_reinj:.15f}")
print(f"   laminar region: [{FP - c:.8f}, {FP + c:.8f}]")

np.random.seed(42)
x = 0.5
for _ in range(500000):
    x = F(x, gamma_reinj)

reinj = []
lam_lo, lam_hi = FP - c, FP + c
in_lam = (lam_lo <= x <= lam_hi)

step = 0
while len(reinj) < TARGET:
    x_prev = x
    x = F(x, gamma_reinj)
    was_in = in_lam
    in_lam = (lam_lo <= x <= lam_hi)
    if in_lam and not was_in:
        reinj.append(x - FP)
    step += 1
    if step > 500_000_000:
        print(f"  WARNING: reached 500M steps with only {len(reinj)} reinjections")
        break

reinj = np.array(reinj)
print(f"  Collected {len(reinj)} reinjections in {step} steps")
print(f"  Range: [{reinj.min():.8f}, {reinj.max():.8f}]")

out_reinj = os.path.join(DATADIR, 'rulkov_reinjection_c=0.01.dat')
header = f"c={c:.6f} eps={EPS_REINJ:.8e} gamma={gamma_reinj:.10f} fp={FP:.15f}"
np.savetxt(out_reinj, reinj, header=header, fmt='%.15e')
print(f"  Saved → {out_reinj}")

# ── 2. M-function analysis ────────────────────────────────────────────
print(f"\n{'='*60}")
print("2. M-function analysis")

x_in = reinj[(reinj >= -c) & (reinj <= c)]
x_sorted = np.sort(x_in)
N = len(x_sorted)

M_vals = np.cumsum(x_sorted) / np.arange(1, N + 1)

# Local map breakpoint: f(-c) = a2*c² + (-c) + eps
xs1_local = A2 * c**2 + (-c) + EPS_REINJ
print(f"  f(-c) = {xs1_local:.8f}")

# Find breakpoints from M-function slope changes
# Use derivative of M to detect transitions
window = max(20, N // 500)
M_smooth = np.convolve(M_vals, np.ones(window)/window, mode='same')

# Fit two branches independently: each gets its own M-function
mask1 = x_sorted < xs1_local
mask2 = x_sorted >= xs1_local

if mask1.sum() > 50 and mask2.sum() > 50:
    # Branch 1: M computed from branch 1 data only
    x_b1 = x_sorted[mask1]
    M_b1 = np.cumsum(x_b1) / np.arange(1, len(x_b1) + 1)
    m1, b1, r1, _, _ = stats.linregress(x_b1, M_b1)

    # Branch 2: M computed fresh from breakpoint
    x_b2 = x_sorted[mask2]
    M_b2 = np.cumsum(x_b2) / np.arange(1, len(x_b2) + 1)
    m2, b2, r2, _, _ = stats.linregress(x_b2, M_b2)

    alpha1 = (2*m1 - 1) / (1 - m1) if abs(1 - m1) > 1e-10 else 0
    alpha2 = (2*m2 - 1) / (1 - m2) if abs(1 - m2) > 1e-10 else 0
    print(f"  Branch 1 [-c, f(-c)]: m={m1:.4f}, α={alpha1:.4f}, R²={r1**2:.6f}, N={len(x_b1)}")
    print(f"  Branch 2 [f(-c), c]:  m={m2:.4f}, α={alpha2:.4f}, R²={r2**2:.6f}, N={len(x_b2)}")
else:
    m_all, b_all, r_all, _, _ = stats.linregress(x_sorted, M_vals)
    m1, m2 = m_all, m_all
    alpha1 = alpha2 = (2*m_all - 1) / (1 - m_all)
    print(f"  Single branch: m={m_all:.4f}, α={alpha1:.4f}")


# ── 3. RPD φ (numerical histogram + theoretical) ─────────────────────
print(f"\n{'='*60}")
print("3. RPD computation")

nbins = 200
counts, edges = np.histogram(x_in, bins=nbins, range=(-c, c))
centers = 0.5 * (edges[:-1] + edges[1:])
dw = edges[1] - edges[0]
phi_num = counts / (counts.sum() * dw)

# Two-branch piecewise RPD (Krause et al.)
# M-function kernel references LEFT boundary of each branch:
#   Branch 1: phi_1(x) = b*k1*(x + c)^alpha1       for x in [-c, x_s1)
#   Branch 2: phi_2(x) = b*k2*(x - x_s1)^alpha2    for x in [x_s1, c]
# alpha1 < 0 => diverges at -c, decays toward x_s1 (matches observed spike)
# Weight ratio k2/k1 set by fraction of reinjections per branch

frac1 = len(x_b1) / N
frac2 = len(x_b2) / N

k1_phi = 1.0
I1_raw = (xs1_local + c)**(alpha1 + 1) / (alpha1 + 1)
I2_raw = (c - xs1_local)**(alpha2 + 1) / (alpha2 + 1)

k2_phi = (frac2 / frac1) * (I1_raw / I2_raw) if I2_raw > 0 else 1.0
b_phi = 1.0 / (k1_phi * I1_raw + k2_phi * I2_raw)

norm_check = b_phi * (k1_phi * I1_raw + k2_phi * I2_raw)
print(f"  normalization: {norm_check:.6f}")
print(f"  b = {b_phi:.4f}, k1 = {k1_phi:.4f}, k2 = {k2_phi:.4f}")
print(f"  Branch 1 events: {len(x_b1)} ({100*frac1:.1f}%)")
print(f"  Branch 2 events: {len(x_b2)} ({100*frac2:.1f}%)")

def phi_theo_pw(x, xs1, a1, a2_rpd, b, k1, k2):
    if x < -c or x > c:
        return 0.0
    if x < xs1:
        dx = x + c
        if dx <= 0:
            dx = 1e-15
        return b * k1 * dx**a1
    else:
        dx = x - xs1
        if dx <= 0:
            dx = 1e-15
        return b * k2 * dx**a2_rpd

# Theoretical curves on fine grids (separate per branch)
x_theo_b1 = np.linspace(-c, xs1_local - 1e-6, 500)
x_theo_b2 = np.linspace(xs1_local + 1e-6, c, 500)
phi_theo_b1 = np.array([phi_theo_pw(x, xs1_local, alpha1, alpha2,
                                     b_phi, k1_phi, k2_phi) for x in x_theo_b1])
phi_theo_b2 = np.array([phi_theo_pw(x, xs1_local, alpha1, alpha2,
                                     b_phi, k1_phi, k2_phi) for x in x_theo_b2])

# ── 4. PDLL psi (numerical histogram + theoretical) ──────────────────
print(f"\n{'='*60}")
print("4. PDLL computation")

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
print(f"  Std: {lam_lengths.std():.1f}")

l_max_plot = min(int(np.percentile(lam_lengths, 99.5)), lam_lengths.max())
nbins_psi = min(100, l_max_plot)
counts_psi, edges_psi = np.histogram(lam_lengths, bins=nbins_psi,
                                      range=(1, l_max_plot), density=True)
centers_psi = 0.5 * (edges_psi[:-1] + edges_psi[1:])

# Theoretical psi: pushforward of phi through l(x,c,eps)
# l(x) = [arctan(c*r) - arctan(x*r)] / sqrt(a2*eps),  r = sqrt(a2/eps)
# dl/dx = -1/(eps + a2*x^2)
# psi(l) = phi(x(l)) * (eps + a2*x(l)^2)

from scipy.interpolate import interp1d

def l_type1(x_entry, c_val, a2_val, eps_val):
    if a2_val * eps_val <= 0:
        return 0.0
    sqae = np.sqrt(a2_val * eps_val)
    ratio = np.sqrt(a2_val / eps_val)
    return (np.arctan(c_val * ratio) - np.arctan(x_entry * ratio)) / sqae

x_entry_grid = np.linspace(-c + 1e-8, c - 1e-8, 50000)
l_of_x = np.array([l_type1(xe, c, A2, EPS_REINJ) for xe in x_entry_grid])

l_grid = np.linspace(1, l_max_plot, 500)
valid = l_of_x > 0
psi_theo = np.zeros_like(l_grid)
l_at_break = 0.0

if valid.sum() > 10:
    l_interp = interp1d(l_of_x[valid][::-1], x_entry_grid[valid][::-1],
                        bounds_error=False, fill_value=np.nan)
    x_of_l = l_interp(l_grid)
    l_at_break = l_type1(xs1_local, c, A2, EPS_REINJ)

    for i, (l_val, x_val) in enumerate(zip(l_grid, x_of_l)):
        if np.isnan(x_val):
            continue
        phi_val = phi_theo_pw(x_val, xs1_local, alpha1, alpha2,
                              b_phi, k1_phi, k2_phi)
        psi_theo[i] = phi_val * (EPS_REINJ + A2 * x_val**2)

    # Split psi theory into two branches at l_at_break
    psi_b1_mask = l_grid >= l_at_break
    psi_b2_mask = l_grid < l_at_break
    l_psi_b1 = l_grid[psi_b1_mask]
    psi_theo_b1 = psi_theo[psi_b1_mask]
    l_psi_b2 = l_grid[psi_b2_mask]
    psi_theo_b2 = psi_theo[psi_b2_mask]
    print(f"  l at breakpoint: {l_at_break:.1f}")
else:
    l_psi_b1 = l_psi_b2 = np.array([])
    psi_theo_b1 = psi_theo_b2 = np.array([])

# ── 5. Scaling analysis ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("5. Scaling analysis")

scaling_file = os.path.join(DATADIR, 'rulkov_scaling_v2_c=0.01.csv')
sdata = np.loadtxt(scaling_file, delimiter=',', comments='#')
log10_eps_s = sdata[:, 0]
eps_s = sdata[:, 1]
mean_l_s = sdata[:, 3]

mask_s = (mean_l_s > 0) & (sdata[:, 6] >= 100)
eps_s = eps_s[mask_s]
mean_l_s = mean_l_s[mask_s]
log10_eps_s = log10_eps_s[mask_s]

log_eps = np.log(eps_s)
log_ml = np.log(mean_l_s)
nu_coeffs = np.polyfit(log_eps, log_ml, 1)
nu_num = -nu_coeffs[0]

def phi_for_integral(x):
    return phi_theo_pw(x, xs1_local, alpha1, alpha2, b_phi, k1_phi, k2_phi)

mean_l_theo = np.zeros(len(eps_s))
for i, ep in enumerate(eps_s):
    def integrand(x, _ep=ep):
        return phi_for_integral(x) * l_type1(x, c, A2, _ep)
    result, _ = quad(integrand, -c, c, limit=500, epsabs=1e-12, epsrel=1e-10)
    mean_l_theo[i] = result

log_ml_th = np.log(mean_l_theo[mean_l_theo > 0])
log_eps_th = log_eps[mean_l_theo > 0]
if len(log_ml_th) > 3:
    nu_th = -np.polyfit(log_eps_th, log_ml_th, 1)[0]
else:
    nu_th = np.nan

print(f"  nu (numerical): {nu_num:.4f}")
print(f"  nu (theoretical): {nu_th:.4f}")

# ======================================================================
# 6. PUBLICATION FIGURES
# ======================================================================
print(f"\n{'='*60}")
print("6. Generating publication figures")

TICK_SIZE = 30
LABEL_SIZE = 39
INSET_TICK = 20
MS = 40
LW_THEO = 3.0

def set_ticks(ax, n=4):
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.locator_params(axis='x', nbins=n)
    ax.locator_params(axis='y', nbins=n)

# ── 6a. M global (cumulative, scatter + theoretical) ─────────────────
fig_m, ax = plt.subplots(figsize=(8, 6))
subsample = max(1, len(x_sorted) // 5000)
ax.scatter(x_sorted[::subsample], M_vals[::subsample], s=MS, c='blue',
           marker='o', edgecolors='none', alpha=0.7)

# Theoretical M(x) = ∫_{-c}^{x} τ·φ(τ)dτ / ∫_{-c}^{x} φ(τ)dτ
x_theo_m = np.linspace(-c + 1e-8, c, 500)
M_theo = np.empty_like(x_theo_m)
for i, xv in enumerate(x_theo_m):
    num_val, _ = quad(lambda t: t * phi_for_integral(t), -c, xv,
                      limit=200, points=[xs1_local])
    den_val, _ = quad(lambda t: phi_for_integral(t), -c, xv,
                      limit=200, points=[xs1_local])
    M_theo[i] = num_val / den_val if abs(den_val) > 1e-30 else np.nan
ax.plot(x_theo_m, M_theo, '-', color='red', lw=LW_THEO, zorder=5)

ax.set_xlabel(r'$\tilde{x}$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$M(\tilde{x})$', fontsize=LABEL_SIZE)
set_ticks(ax)
fig_m.tight_layout()
fig_m.savefig(os.path.join(FIGDIR, 'M_f_c=0.01.png'), dpi=300, bbox_inches='tight')
fig_m.savefig(os.path.join(FIGDIR, 'M_f_c=0.01.pdf'), bbox_inches='tight')
plt.close(fig_m)
print("  Saved M_f_c=0.01")

# ── 6b. M branch 1: scatter + red fit ───────────────────────────────
fig_m1, ax = plt.subplots(figsize=(8, 6))
ss1 = max(1, len(x_b1) // 3000)
ax.scatter(x_b1[::ss1], M_b1[::ss1], s=MS, c='blue',
           marker='o', edgecolors='none', alpha=0.7)
x_fit1 = np.linspace(x_b1[0], x_b1[-1], 200)
ax.plot(x_fit1, m1 * x_fit1 + b1, 'r-', lw=LW_THEO)
ax.set_xlabel(r'$\tilde{x}$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$M_1(\tilde{x})$', fontsize=LABEL_SIZE)
set_ticks(ax)
fig_m1.tight_layout()
fig_m1.savefig(os.path.join(FIGDIR, 'M1_f_c=0.01.png'), dpi=300, bbox_inches='tight')
fig_m1.savefig(os.path.join(FIGDIR, 'M1_f_c=0.01.pdf'), bbox_inches='tight')
plt.close(fig_m1)
print("  Saved M1_f_c=0.01")

# ── 6c. M branch 2: scatter + red fit ───────────────────────────────
fig_m2, ax = plt.subplots(figsize=(8, 6))
ss2 = max(1, len(x_b2) // 3000)
ax.scatter(x_b2[::ss2], M_b2[::ss2], s=MS, c='blue',
           marker='o', edgecolors='none', alpha=0.7)
x_fit2 = np.linspace(x_b2[0], x_b2[-1], 200)
ax.plot(x_fit2, m2 * x_fit2 + b2, 'r-', lw=LW_THEO)
ax.set_xlabel(r'$\tilde{x}$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$M_2(\tilde{x})$', fontsize=LABEL_SIZE)
set_ticks(ax)
fig_m2.tight_layout()
fig_m2.savefig(os.path.join(FIGDIR, 'M2_f_c=0.01.png'), dpi=300, bbox_inches='tight')
fig_m2.savefig(os.path.join(FIGDIR, 'M2_f_c=0.01.pdf'), bbox_inches='tight')
plt.close(fig_m2)
print("  Saved M2_f_c=0.01")

# ── 6d. RPD phi: scatter + red theory (per branch, no connecting) ────
fig_rpd, ax = plt.subplots(figsize=(8, 6))
pos = phi_num > 0
ax.scatter(centers[pos], phi_num[pos], s=MS, c='blue',
           marker='o', edgecolors='none', alpha=0.7, zorder=2)
ax.plot(x_theo_b1, phi_theo_b1, 'r-', lw=LW_THEO, zorder=3)
ax.plot(x_theo_b2, phi_theo_b2, 'r-', lw=LW_THEO, zorder=3)
ax.set_xlabel(r'$\tilde{x}$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$\phi(\tilde{x})$', fontsize=LABEL_SIZE)
ax.set_ylim(0, phi_num[pos].max() * 1.15)
set_ticks(ax)

# Inset: branch 2 flat region with theory
ax_in = ax.inset_axes([0.35, 0.35, 0.55, 0.55])
flat = (centers > xs1_local) & (centers < c) & (phi_num > 0)
ax_in.scatter(centers[flat], phi_num[flat], s=MS, c='blue',
              marker='o', edgecolors='none', alpha=0.7, zorder=2)
ax_in.plot(x_theo_b2, phi_theo_b2, 'r-', lw=LW_THEO, zorder=3)
ax_in.tick_params(labelsize=INSET_TICK)
ax_in.set_xlim(xs1_local - 0.0005, c + 0.0005)
ax_in.set_ylim(0, 24)

fig_rpd.tight_layout()
fig_rpd.savefig(os.path.join(FIGDIR, 'rpd_f_c=0.01.png'), dpi=300, bbox_inches='tight')
fig_rpd.savefig(os.path.join(FIGDIR, 'rpd_f_c=0.01.pdf'), bbox_inches='tight')
plt.close(fig_rpd)
print("  Saved rpd_f_c=0.01")

# ── 6e. PDLL psi: scatter + red theory (per branch) ─────────────────
fig_psi, ax = plt.subplots(figsize=(8, 6))
ax.scatter(centers_psi, counts_psi, s=MS, c='blue',
           marker='o', edgecolors='none', alpha=0.7, zorder=2)
if len(l_psi_b1) > 0:
    ax.plot(l_psi_b1, psi_theo_b1, 'r-', lw=LW_THEO, zorder=3)
if len(l_psi_b2) > 0:
    ax.plot(l_psi_b2, psi_theo_b2, 'r-', lw=LW_THEO, zorder=3)
ax.set_xlabel(r'$l$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$\psi(l)$', fontsize=LABEL_SIZE)
set_ticks(ax)

# Inset: flat body region with theory
ax_in2 = ax.inset_axes([0.15, 0.35, 0.55, 0.55])
flat_psi = counts_psi < 0.05
ax_in2.scatter(centers_psi[flat_psi], counts_psi[flat_psi], s=MS, c='blue',
               marker='o', edgecolors='none', alpha=0.7, zorder=2)
if len(l_psi_b2) > 0:
    body2 = psi_theo_b2 < 0.05
    if body2.any():
        ax_in2.plot(l_psi_b2[body2], psi_theo_b2[body2], 'r-',
                    lw=LW_THEO, zorder=3)
if len(l_psi_b1) > 0:
    body1 = psi_theo_b1 < 0.05
    if body1.any():
        ax_in2.plot(l_psi_b1[body1], psi_theo_b1[body1], 'r-',
                    lw=LW_THEO, zorder=3)
ax_in2.tick_params(labelsize=INSET_TICK)
ax_in2.set_xlim(0, 55)
ax_in2.set_ylim(0, 0.005)

fig_psi.tight_layout()
fig_psi.savefig(os.path.join(FIGDIR, 'pdll_f_c=0.01.png'), dpi=300, bbox_inches='tight')
fig_psi.savefig(os.path.join(FIGDIR, 'pdll_f_c=0.01.pdf'), bbox_inches='tight')
plt.close(fig_psi)
print("  Saved pdll_f_c=0.01")

# ── 6f. Local map fit (manuscript: local_map_fit_c=0.01.png) ────────
print("  Generating local_map_fit_c=0.01")
np.random.seed(42)
x_map = 0.5
for _ in range(500000):
    x_map = F(x_map, gamma_reinj)
map_cx, map_cy = [], []
for _ in range(200000):
    x_prev = x_map
    x_map = F(x_map, gamma_reinj)
    map_cx.append(x_prev - FP)
    map_cy.append(x_map - FP)
map_cx, map_cy = np.array(map_cx), np.array(map_cy)

fig_lm, ax = plt.subplots(figsize=(8, 6))
local_mask = (np.abs(map_cx) < c * 15) & (np.abs(map_cy) < c * 15)
cx_l, cy_l = map_cx[local_mask], map_cy[local_mask]
ss_l = max(1, len(cx_l) // 5000)
ax.scatter(cx_l[::ss_l], cy_l[::ss_l], s=MS, c='blue', alpha=0.5,
           edgecolors='none', zorder=2)
x_anal = np.linspace(-c * 15, c * 15, 1000)
y_anal_full = F(x_anal + FP, gamma_reinj) - FP
ax.plot(x_anal, y_anal_full, 'k-', lw=LW_THEO, zorder=3)
ax.plot([-c*15, c*15], [-c*15, c*15], 'k--', lw=0.8, alpha=0.5)
ax.axvline(-c, color='gray', ls=':', lw=0.8, alpha=0.5)
ax.axvline(c, color='gray', ls=':', lw=0.8, alpha=0.5)
ax.set_xlim(-c * 15, c * 15)
ax.set_ylim(-c * 15, c * 15)
ax.set_xlabel(r'$x_n - x^*$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$x_{n+1} - x^*$', fontsize=LABEL_SIZE)
set_ticks(ax)
fig_lm.tight_layout()
fig_lm.savefig(os.path.join(FIGDIR, 'local_map_fit_c=0.01.png'), dpi=300, bbox_inches='tight')
fig_lm.savefig(os.path.join(FIGDIR, 'local_map_fit_c=0.01.pdf'), bbox_inches='tight')
plt.close(fig_lm)
print("  Saved local_map_fit_c=0.01")

# ── 6g. Scaling: scatter + red theory + fit lines ───────────────────
fig_sc, ax = plt.subplots(figsize=(8, 6))
ax.scatter(log_eps, log_ml, s=MS*1.5, c='blue',
           marker='o', edgecolors='none', alpha=0.8, zorder=2)
if mean_l_theo.min() > 0:
    ax.plot(log_eps_th, log_ml_th, 'r-', lw=LW_THEO, zorder=3)
x_fit_sc = np.linspace(log_eps.min(), log_eps.max(), 100)
ax.plot(x_fit_sc, nu_coeffs[0]*x_fit_sc + nu_coeffs[1],
        'b--', lw=1.5, alpha=0.5, zorder=1)
ax.plot(x_fit_sc, -0.5*x_fit_sc + (log_ml.mean() + 0.5*log_eps.mean()),
        'k:', lw=1.2, alpha=0.4, zorder=1)
ax.set_xlabel(r'$\ln(\varepsilon)$', fontsize=LABEL_SIZE)
ax.set_ylabel(r'$\ln(\langle l \rangle)$', fontsize=LABEL_SIZE)
set_ticks(ax)
fig_sc.tight_layout()
fig_sc.savefig(os.path.join(FIGDIR, 'rulkov_scaling_v2_c=0.01.png'),
               dpi=300, bbox_inches='tight')
fig_sc.savefig(os.path.join(FIGDIR, 'rulkov_scaling_v2_c=0.01.pdf'),
               bbox_inches='tight')
plt.close(fig_sc)
print("  Saved rulkov_scaling_v2_c=0.01")

print(f"\n{'='*60}")
print("DONE. Key results:")
print(f"  x* (correct): {FP:.15f}")
print(f"  Branches: 2 (breakpoint at f(-c) = {xs1_local:.8f})")
print(f"  M slopes: m1={m1:.4f}, m2={m2:.4f}")
print(f"  RPD exponents: a1={alpha1:.4f}, a2={alpha2:.4f}")
print(f"  RPD weights: k1={k1_phi:.4f}, k2={k2_phi:.4f}, b={b_phi:.4f}")
print(f"  nu (numerical): {nu_num:.4f}")
print(f"  nu (theoretical): {nu_th:.4f}")
print(f"  <l> at eps={EPS_REINJ}: {lam_lengths.mean():.1f} (sim)")
print(f"{'='*60}")
