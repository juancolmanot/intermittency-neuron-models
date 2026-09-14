#!/usr/bin/env python3
"""Fig. 1 candidate: bifurcation diagram of the Rulkov fast map F(x)=alpha/(1+x^2)+gamma, alpha=4.8.
Right-to-left continuation sweep (as in the submitted script) + fixed-point branches from the cubic,
solid = stable (|F'|<1), dashed = unstable. Only gamma_SN is marked: at alpha = 4.8 the chaotic
attractor is destroyed at the fold itself (N_u lies inside the chaotic band), so there is no
separate crisis line there. The external (boundary) crisis that does exist at alpha = 4.8 is the one
at gamma_EC = -4.263, where the chaotic attractor grown from the period-doubling cascade of the upper
fixed-point branch (flip at gamma_PD = -4.640) collides with N_u and is destroyed."""
import numpy as np, matplotlib, sys, os
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 14})
ALPHA = 4.8; GAMMA_SN = -2.9314; GAMMA_EC = -4.2633   # boundary crisis of the upper-branch chaotic attractor
F = lambda x, g: ALPHA/(1+x*x) + g

def sweep(g_range, n_gamma=4000, N_settle=1000, N_plot=150, direction='down', x0=0.5):
    gammas = np.linspace(g_range[1], g_range[0], n_gamma) if direction == 'down' else np.linspace(g_range[0], g_range[1], n_gamma)
    x = x0; G=[]; X=[]
    for g in gammas:
        for _ in range(N_settle): x = F(x, g)
        for _ in range(N_plot): x = F(x, g); G.append(g); X.append(x)
    return np.array(G), np.array(X)

def branches(g_range, n=4000):
    gammas = np.linspace(*g_range, n); segs = {'stable': [], 'unstable': []}
    cur = {'stable': [], 'unstable': []}
    for g in gammas:
        r = np.roots([1, -g, 1, -(g+ALPHA)]); r = np.sort(r[np.abs(r.imag) < 1e-9].real)
        for x in r:
            k = 'stable' if abs(-2*ALPHA*x/(1+x*x)**2) < 1 else 'unstable'
            cur[k].append((g, x))
    return cur

g_range = (-5.0, -1.9)
G, X = sweep(g_range)
def sweep_indep(g_range, n_gamma=4000, N_settle=1000, N_plot=150, x0=0.0):
    # independent initial condition x0 at every gamma (no continuation): exposes attractors
    # coexisting with N_s around the upper fixed-point branch
    G=[]; X=[]
    for g in np.linspace(g_range[0], g_range[1], n_gamma):
        x = x0
        for _ in range(N_settle): x = F(x, g)
        for _ in range(N_plot): x = F(x, g); G.append(g); X.append(x)
    return np.array(G), np.array(X)
G2, X2 = sweep_indep(g_range)
br = branches(g_range)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(G, X, '.', color='k', alpha=0.5, ms=0.12, rasterized=True)
ax.plot(G2, X2, '.', color='k', alpha=0.5, ms=0.12, rasterized=True)
for k, st in (('stable', '-'), ('unstable', '--')):
    pts = np.array(br[k])
    # split into contiguous branches: sort by gamma then break where x jumps
    order = np.lexsort((pts[:,1], pts[:,0])); pts = pts[order]
    # group by nearest-neighbour continuity
    used = np.zeros(len(pts), bool)
    for i in range(len(pts)):
        if used[i]: continue
        chain = [i]; used[i] = True; last = pts[i]
        for j in range(i+1, len(pts)):
            if used[j]: continue
            if pts[j,0] - last[0] > 3*(g_range[1]-g_range[0])/4000: break
            if abs(pts[j,1]-last[1]) < 0.05 and pts[j,0] > last[0]:
                chain.append(j); used[j] = True; last = pts[j]
        if len(chain) > 5:
            c = pts[chain]; ax.plot(c[:,0], c[:,1], 'k'+st, lw=2.0)
ax.axvline(GAMMA_SN, color='red', ls='--', lw=1.6, zorder=5)
ax.text(GAMMA_SN + 0.05, -2.45, r'$\gamma_{SN}$', fontsize=18, color='red')
ax.axvline(GAMMA_EC, color='red', ls='--', lw=1.6, zorder=5)
ax.text(GAMMA_EC + 0.05, -2.45, r'$\gamma_{EC}$', fontsize=18, color='red')
ax.text(-4.0, -0.55, r'$N_u$', fontsize=20, fontstyle='italic')
ax.text(-3.55, -2.75, r'$N_s$', fontsize=20, fontstyle='italic')
ax.set_xlabel(r'$\gamma$', fontsize=22); ax.set_ylabel(r'$x$', fontsize=22)
ax.tick_params(labelsize=16); ax.set_xlim(g_range); ax.set_ylim(-3.2, 2.5)
plt.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'Fig1.png')
fig.savefig(out, dpi=200, bbox_inches='tight'); print('saved', out)
