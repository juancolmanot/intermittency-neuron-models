#!/usr/bin/env python3
"""R2-2: scan of the laminar half-width c for the Rulkov map at fixed eps.

For each c the script collects reinjections with cpp/cscan/rulkov_cscan (same
protocol as the shipped generator), then measures
  * P1(c)   : fraction of reinjections landing in the Branch-1 sub-interval [-c, x_s),
              x_s = f(-c) = q c^2 - c + eps  (Eq. xs of the manuscript)
  * the two terms q c^2 and eps that set the sub-interval width x_s + c
  * single-branch M fit (slope m, R^2) and two-branch slopes m1, m2
  * RPD jump ratio phi(x_s^-)/phi(x_s^+) from a fine histogram
Outputs: data/rulkov_cscan/reinj_c=<c>.dat, data/rulkov_cscan_summary.csv,
figures/rulkov_cscan.{png,pdf}.
"""
import os, subprocess, sys
import numpy as np
from scipy import stats

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE    = os.path.join(BASE, 'src', 'rulkov_cscan')
RAWDIR = os.path.join(BASE, 'data', 'rulkov_cscan')
OUTCSV = os.path.join(BASE, 'data', 'rulkov_cscan_summary.csv')
FIGDIR = os.path.join(BASE, 'figures')
os.makedirs(RAWDIR, exist_ok=True)

ALPHA_MAP = 4.8
GAMMA_C   = -2.931400258306671
FP        = -1.765458470442114
A2        = -ALPHA_MAP * (1 - 3*FP**2) / (1 + FP**2)**3      # = 0.5745, Appendix A
EPS       = 3e-4
TARGET    = 200_000
MAXSTEPS  = 5_000_000_000
C_VALUES  = [0.2, 0.15, 0.1, 0.07, 0.05, 0.035, 0.025, 0.02, 0.015, 0.01, 0.007, 0.005]

def collect(c):
    f = os.path.join(RAWDIR, f'reinj_c={c:.4f}.dat')
    if not os.path.isfile(f):
        with open(f, 'w') as fh:
            r = subprocess.run([EXE, f'{c}', f'{EPS}', str(TARGET), str(MAXSTEPS)],
                               stdout=fh, stderr=subprocess.PIPE, text=True, check=True)
        print(r.stderr.strip(), flush=True)
    return np.loadtxt(f, comments='#')

def mfunc(xs):
    xs = np.sort(xs); return xs, np.cumsum(xs) / np.arange(1, len(xs) + 1)

rows = []
for c in C_VALUES:
    r  = collect(c)
    r  = r[(r >= -c) & (r <= c)]
    xs = A2*c**2 - c + EPS
    P1 = np.mean(r < xs)
    x_sorted, M = mfunc(r)
    m, _, rv, _, _ = stats.linregress(x_sorted, M)
    b1 = x_sorted < xs; b2 = ~b1
    m1 = stats.linregress(x_sorted[b1], M[b1]).slope if b1.sum() > 50 else np.nan
    m2 = stats.linregress(x_sorted[b2], M[b2]).slope if b2.sum() > 50 else np.nan
    # RPD jump across x_s: densities in the two bins adjacent to x_s, each of width (x_s+c)/2
    w = (xs + c) / 2
    left  = np.sum((r >= xs - w) & (r <  xs)) / (w * len(r))
    right = np.sum((r >= xs)     & (r <  xs + w)) / (w * len(r))
    jump  = left / right if right > 0 else np.inf
    rows.append(dict(c=c, N=len(r), x_s=xs, ac2=A2*c**2, eps=EPS, width_ratio=(A2*c**2+EPS)/(2*c),
                     P1=P1, m_single=m, R2_single=rv**2, m1=m1, m2=m2, jump=jump))
    print(f"c={c:<6} N={len(r):7d} P1={P1:.4f} m={m:.4f} R2={rv**2:.6f} m1={m1:.3f} m2={m2:.3f} jump={jump:.2f}", flush=True)

import csv
with open(OUTCSV, 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
print("saved", OUTCSV)

# --- analytic model (no free parameters) ------------------------------------
# The landing density of the global map near the fold is c-independent and equals the
# single-branch RPD measured at c = 0.1: phi_g(x) = b (x - x_i)^alpha, with x_i the lowest
# reachable point x_i = F(F(0)) - x* (F(0) = alpha + gamma is the map maximum) and alpha
# from the c = 0.1 fit. For c < |x_i| every landing in [x_i, -c) climbs the channel and
# enters the window inside [-c, x_s), so
#     P1(c) = G(x_s)/G(c) = [ (x_s - x_i) / (c - x_i) ]^(alpha + 1),   x_s = q c^2 - c + eps,
# which also counts direct landings in [-c, x_s). P1 -> 0 once x_s <= x_i (c >~ |x_i|), so the
# transition scale is c_t ~ |x_i|; for c << |x_i|, P1 -> 1 as (1 - 2c/|x_i|)^(alpha+1) ... .
G_R    = GAMMA_C + EPS
X_I    = ALPHA_MAP / (1 + (ALPHA_MAP + G_R)**2) + G_R - FP
r01    = np.loadtxt(os.path.join(RAWDIR, 'reinj_c=0.1000.dat'), comments='#')
r01    = r01[(r01 >= -0.1) & (r01 <= 0.1)]
xs01, M01 = mfunc(r01); m01 = stats.linregress(xs01, M01).slope
ALPHA01 = (2*m01 - 1) / (1 - m01)
cs  = np.array([d['c'] for d in rows]); P1 = np.array([d['P1'] for d in rows])
def P1_model(c):
    # unified: every reinjection below x_s (direct landing in [-c, x_s) or climb through the
    # channel from [x_i, -c)) is counted by the global CDF G(x) ~ (x - x_i)^(alpha+1)
    c = np.asarray(c, float); xs = A2*c**2 - c + EPS
    num = np.clip(xs - X_I, 0, None); return (num / (c - X_I))**(ALPHA01 + 1)
print(f"x_i = F(F(0)) - x* = {X_I:.5f} (measured min at c=0.1: {r01.min():.5f});  alpha(c=0.1) = {ALPHA01:.4f}")
for d in rows: d['P1_model'] = float(P1_model(d['c']))
with open(OUTCSV, 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
for d in rows: print(f"  c={d['c']:<6} P1={d['P1']:.4f}  model={d['P1_model']:.4f}")

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13, 'font.family': 'serif'})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
cf = np.logspace(np.log10(cs.min()), np.log10(cs.max()), 200)
ax[0].semilogx(cs, P1, 'o', color='blue', label='numerical')
ax[0].semilogx(cf, P1_model(cf), '-', color='red', label=r'$[(x_s-x_i)/(c-x_i)]^{\alpha+1}$')
ax[0].axvline(abs(X_I), ls='--', color='gray', label=rf'$c_t=|x_i|={abs(X_I):.3f}$')
ax[0].axvline(0.01, ls=':', color='gray'); ax[0].axvline(0.1, ls=':', color='gray')
ax[0].set_xlabel('$c$'); ax[0].set_ylabel('$P_1$ (fraction in $[-c, x_s)$)'); ax[0].legend(fontsize=10)
ax[0].text(0.03, 0.95, '(a)', transform=ax[0].transAxes, fontweight='bold', va='top')
ax[1].loglog(cf, A2*cf**2, '-', color='blue', label='$q c^2$')
ax[1].axhline(EPS, color='red', label=r'$\varepsilon$')
ax[1].axvline(np.sqrt(EPS/A2), ls='--', color='gray', label=rf'$c^*=\sqrt{{\varepsilon/q}}={np.sqrt(EPS/A2):.3f}$')
ax[1].set_xlabel('$c$'); ax[1].set_ylabel('contribution to $x_s + c$'); ax[1].legend(fontsize=10)
ax[1].text(0.03, 0.95, '(b)', transform=ax[1].transAxes, fontweight='bold', va='top')
R2 = np.array([d['R2_single'] for d in rows]); J = np.array([d['jump'] for d in rows])
ax[2].semilogx(cs, J, 's', color='blue'); ax[2].set_xlabel('$c$'); ax[2].set_ylabel(r'$\phi(x_s^-)/\phi(x_s^+)$', color='blue')
ax2 = ax[2].twinx(); ax2.semilogx(cs, 1 - R2, '^', color='red'); ax2.set_ylabel('$1 - R^2$ (single-branch $M$ fit)', color='red'); ax2.set_yscale('log')
ax[2].text(0.03, 0.95, '(c)', transform=ax[2].transAxes, fontweight='bold', va='top')
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'rulkov_cscan.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(FIGDIR, 'rulkov_cscan.pdf'), bbox_inches='tight')
print("saved figures/rulkov_cscan.{png,pdf}")
