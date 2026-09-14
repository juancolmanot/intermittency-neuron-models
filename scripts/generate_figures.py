#!/usr/bin/env python3
"""Generate all Izhikevich manuscript figures per PROMPT.md spec."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11,
})

DATADIR = "code/datafiles"
OUTDIR = "."
FP = -98.05

THEO_COLOR = 'r'
NUM_COLOR = 'b'

TICK_SIZE = 18


def rk4_izhikevich(d_val, T=6000.0, dt=0.005, v0=-60.0, u0=-98.0):
    a, b, c_reset, I_ext, v_peak = 0.2, 2.0, -56.0, -99.0, 30.0
    N = int(T / dt)
    t = np.arange(N) * dt
    v = np.zeros(N)
    u = np.zeros(N)
    v[0], u[0] = v0, u0

    def fv(v, u): return 0.04*v*v + 5.0*v + 140.0 - u + I_ext
    def fu(v, u): return a*(b*v - u)

    for i in range(N - 1):
        vi, ui = v[i], u[i]
        k1v = fv(vi, ui)
        k1u = fu(vi, ui)
        k2v = fv(vi + 0.5*dt*k1v, ui + 0.5*dt*k1u)
        k2u = fu(vi + 0.5*dt*k1v, ui + 0.5*dt*k1u)
        k3v = fv(vi + 0.5*dt*k2v, ui + 0.5*dt*k2u)
        k3u = fu(vi + 0.5*dt*k2v, ui + 0.5*dt*k2u)
        k4v = fv(vi + dt*k3v, ui + dt*k3u)
        k4u = fu(vi + dt*k3v, ui + dt*k3u)
        v[i+1] = vi + (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
        u[i+1] = ui + (dt/6.0)*(k1u + 2*k2u + 2*k3u + k4u)
        if v[i+1] >= v_peak:
            v[i] = v_peak
            v[i+1] = c_reset
            u[i+1] += d_val
    return t, v, u


def set_tick_params(ax, nticks=3):
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.locator_params(axis='x', nbins=nticks)
    ax.locator_params(axis='y', nbins=nticks)


# ── Fig 15: Regular spiking (limit cycle) ──────────────────────────────
def _plot_with_resets(ax, x, y, reset_idx, color='b', lw=1.5):
    """Plot continuous segments as solid, reset jumps as dashed."""
    cuts = sorted(set(reset_idx))
    prev = 0
    for ci in cuts:
        if ci > prev:
            ax.plot(x[prev:ci+1], y[prev:ci+1], color + '-', lw=lw)
        if ci + 1 < len(x):
            ax.plot(x[ci:ci+2], y[ci:ci+2], color='gray', ls='--', lw=1.0, alpha=0.7)
        prev = ci + 1
    if prev < len(x):
        ax.plot(x[prev:], y[prev:], color + '-', lw=lw)


def fig_regular_spiking():
    t, v, u = rk4_izhikevich(d_val=-11.0, T=15000.0, dt=0.001)
    # Time series: show well past transient
    mask = (t > 14000) & (t <= 14200)
    tm, vm, um = t[mask], v[mask], u[mask]

    # Detect reset indices: v jumps from 30 to -56
    reset_idx = []
    for i in range(len(vm) - 1):
        if vm[i] >= 29.0 and vm[i+1] < -50:
            reset_idx.append(i)

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7),
                             gridspec_kw={'height_ratios': [1, 1, 2]})

    _plot_with_resets(axes[0], tm, vm, reset_idx, NUM_COLOR, 1.5)
    axes[0].set_ylabel(r'$v$ (mV)', fontsize=24)
    axes[0].set_xlabel(r'$t$ (ms)', fontsize=24)
    axes[0].set_title('(a)', loc='left', fontsize=26)
    set_tick_params(axes[0])

    _plot_with_resets(axes[1], tm, um, reset_idx, NUM_COLOR, 1.5)
    axes[1].set_ylabel(r'$u$', fontsize=24)
    axes[1].set_xlabel(r'$t$ (ms)', fontsize=24)
    axes[1].set_title('(b)', loc='left', fontsize=26)
    set_tick_params(axes[1])

    # Phase plane — only last 200ms for clean single-orbit plot
    mask2 = t > 14800
    v2, u2 = v[mask2], u[mask2]
    reset2 = []
    for i in range(len(v2) - 1):
        if v2[i] >= 29.0 and v2[i+1] < -50:
            reset2.append(i)
    _plot_with_resets(axes[2], v2, u2, reset2, NUM_COLOR, 0.8)
    axes[2].set_xlabel(r'$v$ (mV)', fontsize=24)
    axes[2].set_ylabel(r'$u$', fontsize=24)
    axes[2].set_title('(c)', loc='left', fontsize=26)
    set_tick_params(axes[2])

    plt.tight_layout()
    out = os.path.join(OUTDIR, "izhikevich_regular_spiking.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ── Fig 16: Bifurcation diagram + Lyapunov ──────────────────────────────
def fig_bifurcation_lyapunov():
    bif = np.loadtxt(os.path.join(DATADIR, "izhikevich_bifurcation_state_d_u_1.dat"))
    lyap = np.loadtxt(os.path.join(DATADIR, "lyapunov_bifurcation_izhikevich_d_1.dat"))

    d_bif, u_bif = bif[:, 0], bif[:, 1]
    d_lyap, lam1, lam2 = lyap[:, 0], lyap[:, 1], lyap[:, 2]

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6), sharex=True)

    axes[0].scatter(d_bif, u_bif, s=0.05, c=NUM_COLOR, alpha=0.3, rasterized=True)
    axes[0].set_ylabel(r'$u_i$', fontsize=24)
    axes[0].set_title('(a)', loc='left', fontsize=18)
    axes[0].set_xlim(-17, -5)
    set_tick_params(axes[0], nticks=4)

    axes[1].plot(d_lyap, lam1, NUM_COLOR + '-', lw=1.2, label=r'$\lambda_1$')
    axes[1].plot(d_lyap, lam2, 'r-', lw=1.2, label=r'$\lambda_2$')
    axes[1].axhline(0, color='k', ls=':', lw=0.5)
    axes[1].set_xlabel(r'$d$', fontsize=22)
    axes[1].set_ylabel(r'$\lambda_j$', fontsize=22)
    axes[1].set_title('(b)', loc='left', fontsize=18)
    axes[1].legend(fontsize=12)
    set_tick_params(axes[1], nticks=4)

    plt.tight_layout()
    out = os.path.join(OUTDIR, "izhikevich_bifurcation_lyapunov.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ── Fig 17: Chaotic behavior (d=-16): v(t), phase plane, return map ─────
def fig_chaotic_behavior():
    t, v, u = rk4_izhikevich(d_val=-16.0, T=6000.0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) v(t)
    mask = (t > 5000) & (t < 5200)
    axes[0].plot(t[mask], v[mask], NUM_COLOR + '-', lw=1.2)
    axes[0].set_xlabel(r'$t$ (ms)', fontsize=22)
    axes[0].set_ylabel(r'$v$ (mV)', fontsize=22)
    axes[0].set_title('(a)', loc='left', fontsize=18)
    set_tick_params(axes[0])

    # (b) Phase plane (v, u) with nullclines
    mask2 = t > 3000
    axes[1].plot(v[mask2], u[mask2], NUM_COLOR + '-', lw=0.75, alpha=0.3)
    vn = np.linspace(-80, 40, 300)
    un_v = 0.04*vn**2 + 5*vn + 140 - 99
    un_u = 2.0 * vn
    axes[1].plot(vn, un_v, 'k--', lw=2, alpha=0.6, label=r"$\dot{v}=0$")
    axes[1].plot(vn, un_u, 'k:', lw=2, alpha=0.6, label=r"$\dot{u}=0$")
    axes[1].set_xlabel(r'$v$ (mV)', fontsize=22)
    axes[1].set_ylabel(r'$u$', fontsize=22)
    axes[1].set_title('(b)', loc='left', fontsize=18)
    axes[1].legend(fontsize=10, loc='upper right')
    axes[1].set_ylim(-130, -85)
    set_tick_params(axes[1])

    # (c) Return map u_{i+1} vs u_i from data
    mapf = os.path.join(DATADIR, "izhikevich_poincare_map_v=30_d=-16.dat")
    mdata = np.loadtxt(mapf)
    ui, ui1 = mdata[:, 0], mdata[:, 1]
    idx = np.argsort(ui)
    axes[2].plot(ui[idx], ui1[idx], NUM_COLOR + '-', lw=2, alpha=1)
    lims = [-105, -90]
    axes[2].plot(lims, lims, 'k--', lw=0.8, alpha=0.75)
    axes[2].set_xlabel(r'$u_i$', fontsize=22)
    axes[2].set_ylabel(r'$u_{i+1}$', fontsize=22)
    axes[2].set_title('(c)', loc='left', fontsize=18)
    axes[2].set_xlim(-105, -90)
    axes[2].set_ylim(-110, -75)
    set_tick_params(axes[2])

    plt.tight_layout()
    out = os.path.join(OUTDIR, "izhikevich_chaotic_behavior.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ── Fig 18: Time series v(t) + phase plane for d=-11,-12,-13,-16 ───────
def fig_time_series_phase():
    d_vals = [-11, -12.5, -13, -16]

    fig = plt.figure(figsize=(14, 12))
    outer = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    for idx, d in enumerate(d_vals):
        row, col = divmod(idx, 2)
        inner = outer[row, col].subgridspec(2, 1, height_ratios=[0.3, 0.7], hspace=0.05)
        ax_ts = fig.add_subplot(inner[0])
        ax_ph = fig.add_subplot(inner[1])

        T_sim = 20000.0 if d == -11 else 6000.0
        dt_sim = 0.001 if d == -11 else 0.005
        t, v, u = rk4_izhikevich(d_val=float(d), T=T_sim, dt=dt_sim)

        # Time series (top 30%) — wider window to show intermittent alternation
        t_start = T_sim - 1000
        mask = (t > t_start) & (t < t_start + 400)
        ax_ts.plot(t[mask], v[mask], NUM_COLOR + '-', lw=2)
        ax_ts.set_ylabel(r'$v$ (mV)', fontsize=26)
        ax_ts.set_title(f'({chr(97+idx)}) $d = {d}$', loc='left', fontsize=24)
        ax_ts.tick_params(labelbottom=False, axis='both', labelsize=22)
        ax_ts.locator_params(axis='x', nbins=3)
        ax_ts.locator_params(axis='y', nbins=3)

        # Phase plane (bottom 70%) — for periodic d=-11, discard longer transient
        trans_cut = 18000.0 if d == -11 else 3000.0
        mask2 = t > trans_cut
        ax_ph.plot(v[mask2], u[mask2], NUM_COLOR + '-', lw=0.6, alpha=0.4)
        ax_ph.set_xlabel(r'$v$ (mV)', fontsize=26)
        ax_ph.set_ylabel(r'$u$', fontsize=26)
        ax_ph.tick_params(axis='both', labelsize=22)
        ax_ph.locator_params(axis='x', nbins=3)
        ax_ph.locator_params(axis='y', nbins=3)

    out = os.path.join(OUTDIR, "izhikevich_time_series_phase.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ── Fig 19: u_i time series + (u_i, u_{i+2}) maps for d=-11,-12,-12.5,-13 ─
def fig_poincare_maps():
    d_vals = [-11, -12, -12.5, -13]
    d_fnames = {
        -11:   "poincare_fig19_d=-11.dat",
        -12:   "poincare_fig19_d=-12.dat",
        -12.5: "poincare_fig19_d=-12.5.dat",
        -13:   "poincare_fig19_d=-13.dat",
    }

    fig = plt.figure(figsize=(14, 12))
    outer = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    for idx, d in enumerate(d_vals):
        row, col = divmod(idx, 2)
        inner = outer[row, col].subgridspec(2, 1, height_ratios=[0.3, 0.7], hspace=0.05)
        ax_ts = fig.add_subplot(inner[0])
        ax_map = fig.add_subplot(inner[1])

        mapf = os.path.join(DATADIR, d_fnames[d])
        mdata = np.loadtxt(mapf)
        # 3-col: u_{i-2}, u_{i-1}, u_i  →  F² map: col0 vs col2
        ui = mdata[:, 0]
        ui1 = mdata[:, 1]
        ui2 = mdata[:, 2]

        # Time series: use col1 (u_{i-1}) sequence, take first 300 points
        ts_plot = ui1[:300]
        ax_ts.plot(np.arange(len(ts_plot)), ts_plot, NUM_COLOR + '-', lw=1.2)
        ax_ts.set_ylabel(r'$u_i$', fontsize=26)
        ax_ts.set_title(f'({chr(97+idx)}) $d = {d}$', loc='left', fontsize=24)
        ax_ts.tick_params(labelbottom=False, axis='both', labelsize=22)
        ax_ts.locator_params(axis='x', nbins=3)
        ax_ts.locator_params(axis='y', nbins=3)

        # F² map: sort ALL data globally by u_i (col0) to get clean curve
        sort_idx = np.argsort(ui)
        ax_map.plot(ui[sort_idx], ui2[sort_idx], NUM_COLOR + '-', lw=1.2, alpha=1)
        lims_lo = min(ui.min(), ui2.min()) - 0.5
        lims_hi = max(ui.max(), ui2.max()) + 0.5
        ax_map.plot([lims_lo, lims_hi], [lims_lo, lims_hi], 'k--', lw=1, alpha=0.5)
        ax_map.set_xlabel(r'$u_i$', fontsize=26)
        ax_map.set_ylabel(r'$u_{i+2}$', fontsize=26)
        if (idx == 0):
            xlims = (-103, -90)
            ylims = (-105, -85)

        elif (idx == 1):
            xlims = (-101.5, -93)
            ylims = (-105, -75)

        elif (idx == 2):
            xlims = (-101, -94)
            ylims = (-105, -75)

        elif (idx == 3):
            xlims = (-100.2, -94.5)
            ylims = (-105, -75)

        ax_map.set_xlim(xlims)
        ax_map.set_ylim(ylims)
        ax_map.tick_params(axis='both', labelsize=22)
        ax_map.locator_params(axis='x', nbins=3)
        ax_map.locator_params(axis='y', nbins=3)

    out = os.path.join(OUTDIR, "izhikevich_poincare_maps_multi.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved {out}")

# ── Fig 20: Zoomed local maps near fixed point + cubic fits ─────────────
def fig_local_maps_zoom():
    d_vals_zoom = [-11, -12, -12.5, -13]
    labels = [r'$d=-11$', r'$d=-12$', r'$d=-12.5$', r'$d=-13$']
    d_fnames = {
        -11:   "poincare_fig19_d=-11.dat",
        -12:   "poincare_fig19_d=-12.dat",
        -12.5: "poincare_fig19_d=-12.5.dat",
        -13:   "poincare_fig19_d=-13.dat",
    }
    # Fig 19 xlims (absolute coords) per d — used to define zoom window
    fig19_xlims = {
        -11:   (-103, -90),
        -12:   (-101.5, -93),
        -12.5: (-101, -94),
        -13:   (-100.2, -94.5),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (d, label) in enumerate(zip(d_vals_zoom, labels)):
        fname = os.path.join(DATADIR, d_fnames[d])
        mdata = np.loadtxt(fname)
        ui_abs = mdata[:, 0]
        ui2_abs = mdata[:, 2]

        # First pass: rough shift by FP to find actual fixed point
        ui_rough = ui_abs - FP
        ui2_rough = ui2_abs - FP

        # Fit window from fig19 xlims, shrunk 35% per side
        xlo_abs, xhi_abs = fig19_xlims[d]
        xlo_r, xhi_r = xlo_abs - FP, xhi_abs - FP
        xrange = xhi_r - xlo_r
        fit_lo_r = xlo_r + 0.35 * xrange
        fit_hi_r = xhi_r - 0.35 * xrange

        # Fit cubic in rough coords to find crossing with bisector
        mask_r = (ui_rough >= fit_lo_r) & (ui_rough <= fit_hi_r)
        xr = ui_rough[mask_r]
        yr = ui2_rough[mask_r]
        coeffs_r = np.polyfit(xr, yr, 3)
        A, B, C, D = coeffs_r
        fp_roots = np.roots([A, B, C - 1, D])
        fp_real = fp_roots[np.isreal(fp_roots)].real
        fp_abs = FP + fp_real[np.argmin(np.abs(fp_real))]
        print(f"    d={d}: fixed point at u_abs={fp_abs:.4f} (shift from FP={fp_abs-FP:.4f})")

        # Second pass: shift by actual fixed point so crossing is at 0
        ui_s = ui_abs - fp_abs
        ui2_s = ui2_abs - fp_abs

        # Re-fit in corrected coords
        fit_range = fit_hi_r - fit_lo_r
        fit_lo = fit_lo_r - (fp_abs - FP)
        fit_hi = fit_hi_r - (fp_abs - FP)
        mask = (ui_s >= fit_lo) & (ui_s <= fit_hi)
        x = ui_s[mask]
        y = ui2_s[mask]

        coeffs = np.polyfit(x, y, 3)
        A, B, C, D = coeffs

        # Plot window: wider than fit to show cubic curvature
        half_w = fit_range * 0.5 * 1.5
        zoom_lo = -half_w
        zoom_hi = half_w

        plot_mask = (ui_s >= zoom_lo) & (ui_s <= zoom_hi)
        xp = ui_s[plot_mask]
        yp = ui2_s[plot_mask]
        pi = np.argsort(xp)
        axes[i].scatter(xp[pi], yp[pi], s=3, c=NUM_COLOR, alpha=0.7, rasterized=True)

        xfit = np.linspace(zoom_lo, zoom_hi, 500)
        yfit = np.polyval(coeffs, xfit)
        axes[i].plot(xfit, yfit, THEO_COLOR + '-', lw=2.5)
        axes[i].plot([zoom_lo, zoom_hi], [zoom_lo, zoom_hi], 'k--', lw=1.5, alpha=0.5)
        axes[i].set_xlim(zoom_lo, zoom_hi)
        axes[i].set_ylim(zoom_lo, zoom_hi)
        axes[i].set_xlabel(r"$\bar{u}_i$", fontsize=28)
        axes[i].set_ylabel(r"$\bar{u}_{i+2}$", fontsize=28)
        axes[i].set_title(f'({chr(97+i)}) {label}', loc='left', fontsize=24)
        axes[i].tick_params(axis='both', labelsize=22)
        axes[i].locator_params(axis='x', nbins=3)
        axes[i].locator_params(axis='y', nbins=3)

    plt.tight_layout()
    out = os.path.join(OUTDIR, "izhikevich_local_maps_zoom.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ── Fig 21: M-function global + branches ────────────────────────────────
def fig_M_function():
    # Global M
    f_m = os.path.join(DATADIR, "izhikevich_m_function_d=-11.78_c=1.5_n2.dat")
    dm = np.loadtxt(f_m)
    u_m, M_m = dm[:, 0], dm[:, 1]

    # Analytical M
    f_ma = os.path.join(DATADIR, "izhikevich_m_function_analytic_d=-11.78_c=1.5_n2_NEW.dat")
    has_ana = os.path.exists(f_ma)

    # Branches
    f1 = os.path.join(DATADIR, "izhikevich_m1_function_d=-11.78_n2.dat")
    f2 = os.path.join(DATADIR, "izhikevich_m2_function_d=-11.78_n2.dat")
    f3 = os.path.join(DATADIR, "izhikevich_m3_function_d=-11.78_n2.dat")
    d1 = np.loadtxt(f1)
    d2 = np.loadtxt(f2)
    d3 = np.loadtxt(f3)
    u1, m1 = d1[:, 2], d1[:, 3]
    u2, m2 = d2[:, 0], d2[:, 1]
    u3, m3 = d3[:, 0], d3[:, 1]

    uc2, uc3 = -0.835, 0.575

    # (a) Global M as standalone figure
    fig_a, ax_a = plt.subplots(1, 1, figsize=(6.5, 4.5))
    ax_a.scatter(u_m, M_m, s=1, c=NUM_COLOR, alpha=0.5, rasterized=True, label='Numerical')
    if has_ana:
        dma = np.loadtxt(f_ma)
        u_ma, M_ma = dma[:, 0], dma[:, 1]
        ax_a.plot(u_ma, M_ma, THEO_COLOR + '-', lw=2, label='Theoretical')
    ax_a.axvline(uc2, color='gray', ls='--', lw=0.8)
    ax_a.axvline(uc3, color='gray', ls='--', lw=0.8)
    ax_a.set_xlabel(r"$\bar{u}$", fontsize=22)
    ax_a.set_ylabel(r"$M(\bar{u})$", fontsize=22)
    ax_a.legend(fontsize=16)
    set_tick_params(ax_a)
    fig_a.tight_layout()
    out_a = os.path.join(OUTDIR, "izhikevich_M_global.png")
    fig_a.savefig(out_a, dpi=300, bbox_inches='tight')
    plt.close(fig_a)
    print(f"  Saved {out_a}")

    # (b-d) Three branches together
    fig_b, axes_b = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, ub, mb, ylabel, title in zip(
        axes_b,
        [u1, u2, u3], [m1, m2, m3],
        [r'$M_1$', r'$M_2$', r'$M_3$'],
        ['(a)', '(b)', '(c)']
    ):
        slope, intercept = np.polyfit(ub, mb, 1)
        ax.scatter(ub, mb, s=1, c=NUM_COLOR, alpha=0.5, rasterized=True)
        ax.plot(ub, slope * ub + intercept, THEO_COLOR + '-', lw=2,
                label=f'$m={slope:.3f}$')
        ax.set_xlabel(r"$\bar{u}$", fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)
        ax.legend(fontsize=18)
        ax.set_title(title, loc='left', fontsize=20)
        set_tick_params(ax)

    fig_b.tight_layout()
    out_b = os.path.join(OUTDIR, "izhikevich_M_branches.png")
    fig_b.savefig(out_b, dpi=300, bbox_inches='tight')
    plt.close(fig_b)
    print(f"  Saved {out_b}")

    # Also save combined version for backward compat
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].scatter(u_m, M_m, s=1, c=NUM_COLOR, alpha=0.5, rasterized=True, label='Numerical')
    if has_ana:
        dma2 = np.loadtxt(f_ma)
        u_ma2, M_ma2 = dma2[:, 0], dma2[:, 1]
        axes[0, 0].plot(u_ma2, M_ma2, THEO_COLOR + '-', lw=2, label='Theoretical')
    axes[0, 0].axvline(uc2, color='gray', ls='--', lw=0.8)
    axes[0, 0].axvline(uc3, color='gray', ls='--', lw=0.8)
    axes[0, 0].set_xlabel(r"$\bar{u}$", fontsize=22)
    axes[0, 0].set_ylabel(r"$M(\bar{u})$", fontsize=22)
    axes[0, 0].legend(fontsize=16)
    axes[0, 0].set_title('(a)', loc='left', fontsize=20)
    set_tick_params(axes[0, 0])

    for ax, ub, mb, ylabel, title in zip(
        [axes[0, 1], axes[1, 0], axes[1, 1]],
        [u1, u2, u3], [m1, m2, m3],
        [r'$M_1$', r'$M_2$', r'$M_3$'],
        ['(b)', '(c)', '(d)']
    ):
        slope, intercept = np.polyfit(ub, mb, 1)
        ax.scatter(ub, mb, s=1, c=NUM_COLOR, alpha=0.5, rasterized=True)
        ax.plot(ub, slope * ub + intercept, THEO_COLOR + '-', lw=2,
                label=f'$m={slope:.3f}$')
        ax.set_xlabel(r"$\bar{u}$", fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)
        ax.legend(fontsize=18)
        ax.set_title(title, loc='left', fontsize=20)
        set_tick_params(ax)

    plt.tight_layout()
    out = os.path.join(OUTDIR, "izhikevich_M_function_full.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ── Fig 22/23: RPD numerical vs theoretical ─────────────────────────────
def fig_rpd():
    f_num = os.path.join(DATADIR, "izhikevich_rpd_d=-11.78_c=2_n2.dat")
    d_num = np.loadtxt(f_num, comments='#')
    u_num, phi_num = d_num[:, 0], d_num[:, 1]

    f_ana = os.path.join(DATADIR, "izhikevich_rpd_function_analytic_d=-11.78_c=1.5_n2_NEW.dat")
    has_ana = os.path.exists(f_ana)

    uc2, uc3 = -0.835, 0.575

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    mask = phi_num > 0
    ax.scatter(u_num[mask], phi_num[mask], s=4, c=NUM_COLOR, alpha=1,
               label='Numerical', rasterized=True)

    if has_ana:
        d_ana = np.loadtxt(f_ana)
        u_ana, phi_ana = d_ana[:, 0], d_ana[:, 1]

        phi_cap = 2.45
        margin = 0.0085
        regions = [
            (u_ana < (uc2 - margin), 'Theoretical'),
            ((u_ana >= (uc2 + margin)) & (u_ana <= (uc3 - margin)), None),
            (u_ana > (uc3 + margin), None),
        ]
        for rmask, lbl in regions:
            ur = u_ana[rmask]
            pr = phi_ana[rmask]
            keep = (pr > 0) & (pr < phi_cap)
            if keep.sum() < 2:
                continue
            ax.plot(ur[keep], pr[keep], THEO_COLOR + '-', lw=2, label=lbl)

    ax.axvline(uc2, color='gray', ls='--', lw=0.8)
    ax.axvline(uc3, color='gray', ls='--', lw=0.8)
    ax.set_xlabel(r"$\bar{u}$", fontsize=22)
    ax.set_ylabel(r"$\phi(\bar{u})$", fontsize=22)
    ax.set_ylim(bottom=-0.05, top=2.5)
    ax.legend(fontsize=14)
    set_tick_params(ax)
    plt.tight_layout()
    out = os.path.join(OUTDIR, "izhikevich_rpd_comparison.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ── Main ─────────────────────────────────────────────────────────────────
# ── Fig: Wide-view return map showing full multi-branch structure ─────
def fig_return_map_wide():
    d_vals = [-12, -12.5, -13]
    labels = [r'$d=-12$', r'$d=-12.5$', r'$d=-13$']
    d_fnames = {
        -12:   "poincare_fig19_d=-12.dat",
        -12.5: "poincare_fig19_d=-12.5.dat",
        -13:   "poincare_fig19_d=-13.dat",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for i, (d, label) in enumerate(zip(d_vals, labels)):
        fname = os.path.join(DATADIR, d_fnames[d])
        mdata = np.loadtxt(fname)
        ui = mdata[:, 0]
        ui2 = mdata[:, 2]

        idx = np.argsort(ui)
        axes[i].plot(ui[idx], ui2[idx], NUM_COLOR + '-', lw=1.8, alpha=0.8, rasterized=True)
        lo, hi = ui.min() - 1, ui.max() + 1
        axes[i].plot([lo, hi], [lo, hi], 'k--', lw=1.5, alpha=0.6)
        axes[i].set_xlim(lo, hi)
        axes[i].set_ylim(lo, hi)
        axes[i].set_xlabel(r"$u_i$", fontsize=28)
        axes[i].set_ylabel(r"$u_{i+2}$", fontsize=28)
        axes[i].set_title(f'({chr(97+i)}) {label}', loc='left', fontsize=24)
        axes[i].tick_params(axis='both', labelsize=22)
        axes[i].locator_params(axis='x', nbins=4)
        axes[i].locator_params(axis='y', nbins=4)
        axes[i].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    out = os.path.join(OUTDIR, "izhikevich_return_map_wide.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Generating manuscript figures...")
    print("\n1/9: Regular spiking (Fig 15)")
    fig_regular_spiking()
    print("\n2/9: Bifurcation + Lyapunov (Fig 16)")
    fig_bifurcation_lyapunov()
    print("\n3/9: Chaotic behavior d=-16 (Fig 17)")
    fig_chaotic_behavior()
    print("\n4/9: Time series + phase planes (Fig 18)")
    fig_time_series_phase()
    print("\n5/9: Poincaré maps (Fig 19)")
    fig_poincare_maps()
    print("\n6/9: Local maps zoom + cubic fits (Fig 20)")
    fig_local_maps_zoom()
    print("\n7/9: M-function (Fig 21)")
    fig_M_function()
    print("\n8/9: RPD comparison (Fig 22/23)")
    fig_rpd()
    print("\n9/9: Wide-view return map")
    fig_return_map_wide()
    print("\nDone.")
