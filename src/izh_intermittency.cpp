/**
 * izh_intermittency.cpp  — TRUSTED in-project engine (homogenized)
 *
 * Izhikevich intermittency-regime simulator for the scaling / PDLL analysis.
 * Model (verified regime, manuscript §2 / SCALING-IZHIKEVICH.md):
 *     dv/dt = 0.04 v^2 + 5 v + 140 - u + I
 *     du/dt = a (b v - u)
 *     reset: v>=v_th  ->  v<-c_reset,  u<-u+d
 *   a=0.2, b=2.0, c_reset=-56, I=-99, v_th=30.  d = control parameter.
 *
 * Poincare section: u recorded at each UPWARD crossing v=v_th, linearly
 * interpolated to v=v_th (sub-step) so the return map is not biased by
 * integrator overshoot.  Shifted coord: ubar = u_cross - fp, fp=-98.05
 * (manuscript single common shift).
 *
 * Laminar detection (matches dual_clam_scaling.sim_mean_laminar, 2-step
 * look-back on the full crossing sequence; l counts second-iterate steps):
 *   START when |ubar_k|<=clam AND |ubar_{k-2}|>clam
 *   END   when |ubar_{k-2}|<=clam AND |ubar_k|>clam
 *   l_spikes = k_end - k_start ;  l_iter = (l_spikes+1)/2
 *   reinjection point = ubar at first in-window crossing.
 * First `skip_spikes` crossings after transient are discarded (warm-up).
 *
 * Two modes:
 *   --mode mapdump   one long trajectory; dumps (u_n, u_{n+2}) second-iterate
 *                    pairs after transient -> Python fits cubic F^2 = A x^3 +
 *                    B x^2 + C x + D and locates C(d)=1 (d_c), A(d).
 *   --mode episodes  many independent trials; detects laminar episodes at a
 *                    LIST of clam values in one pass; writes per-episode
 *                    (reinj, l_iter) files + one summary line per clam.
 *
 * Deterministic: RNG seeded from --seed; per-trial stream seed+trial.
 * Build: see Makefile (needs boost odeint, OpenMP).
 */
#include <boost/numeric/odeint.hpp>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using state_type = std::array<double, 2>;
namespace odeint = boost::numeric::odeint;

// ---- fixed model constants (verified regime) --------------------------------
static constexpr double A_PARAM = 0.2;
static constexpr double B_PARAM = 2.0;
static constexpr double C_RESET = -56.0;
static constexpr double I_EXT   = -99.0;
static constexpr double V_TH    = 30.0;
static constexpr double FP_SHIFT = -98.05;   // manuscript common shift u*
static constexpr double ATOL = 1e-9;
static constexpr double RTOL = 1e-11;

struct RHS {
    double d_unused; // model RHS does not depend on d (d only in reset)
    void operator()(const state_type &x, state_type &dxdt, double) const {
        const double v = x[0], u = x[1];
        dxdt[0] = 0.04 * v * v + 5.0 * v + 140.0 - u + I_EXT;
        dxdt[1] = A_PARAM * (B_PARAM * v - u);
    }
};

// Integrate one trajectory, returning the sequence of interpolated u-crossings
// (Poincare section v=V_TH), for t in (t_transient, t_transient+t_run].
// `d` is the reset increment (control parameter).
static std::vector<double> crossing_sequence(double d, double v0, double u0,
                                              double t_transient, double t_run,
                                              double dt_max) {
    std::vector<double> crossings;
    crossings.reserve(1 << 15);
    auto stepper = odeint::make_controlled(ATOL, RTOL,
                     odeint::runge_kutta_cash_karp54<state_type>());
    RHS rhs{d};
    state_type x{v0, u0};
    double t = 0.0;
    const double t_end = t_transient + t_run;
    while (t < t_end) {
        state_type x_pre = x;
        double dt = dt_max;
        odeint::controlled_step_result res;
        do { res = stepper.try_step(rhs, x, t, dt); } while (res == odeint::fail);
        // upward crossing of V_TH within the accepted step?
        if (x_pre[0] < V_TH && x[0] >= V_TH) {
            const double dv = x[0] - x_pre[0];
            const double frac = (dv > 0.0) ? (V_TH - x_pre[0]) / dv : 1.0;
            const double u_cross = x_pre[1] + frac * (x[1] - x_pre[1]);
            if (t >= t_transient) crossings.push_back(u_cross);
            // reset FROM the interpolated crossing (spike fires at v=V_TH, u=u_cross)
            // so the Poincare map is a clean function of the section coordinate,
            // not the integrator overshoot.
            x[0] = C_RESET;
            x[1] = u_cross + d;
        }
    }
    return crossings;
}

// ---- largest Lyapunov exponent via variational flow + saltation matrix ------
// 4D state [v,u,w0,w1]; tangent obeys dw/dt = J(v) w, J=[[0.08v+5,-1],[ab,-a]].
// At an upward crossing v=V_TH (event h=v-V_TH, reset R(v,u)=(c_reset,u+d)):
//   S = D_R + (f+ - D_R f-) grad(h)^T / (grad(h)^T f-)
// with D_R=[[0,0],[0,1]], grad(h)=(1,0).  Gives (u-=u at crossing, u+=u-+d):
//   denom = f_v- ,  S = [[f_v+/denom, 0], [(f_u+ - f_u-)/denom, 1]].
using state4 = std::array<double, 4>;
struct RHS4 {
    void operator()(const state4 &x, state4 &dx, double) const {
        const double v = x[0], u = x[1], w0 = x[2], w1 = x[3];
        dx[0] = 0.04 * v * v + 5.0 * v + 140.0 - u + I_EXT;
        dx[1] = A_PARAM * (B_PARAM * v - u);
        const double j00 = 0.08 * v + 5.0, j01 = -1.0;
        const double j10 = A_PARAM * B_PARAM, j11 = -A_PARAM;
        dx[2] = j00 * w0 + j01 * w1;
        dx[3] = j10 * w0 + j11 * w1;
    }
};

static double lyapunov_max(double d, double v0, double u0,
                           double t_transient, double t_run, double dt_max) {
    auto stepper = odeint::make_controlled(ATOL, RTOL,
                     odeint::runge_kutta_cash_karp54<state4>());
    RHS4 rhs;
    state4 x{v0, u0, 1.0, 0.0};
    // normalize tangent
    double nrm = std::hypot(x[2], x[3]); x[2] /= nrm; x[3] /= nrm;
    double t = 0.0;
    const double t_end = t_transient + t_run;
    double log_sum = 0.0;
    double t_acc_start = -1.0;   // time accumulation begins (after transient)
    while (t < t_end) {
        state4 x_pre = x;
        double dt = dt_max;
        odeint::controlled_step_result res;
        do { res = stepper.try_step(rhs, x, t, dt); } while (res == odeint::fail);
        if (x_pre[0] < V_TH && x[0] >= V_TH) {
            const double dv = x[0] - x_pre[0];
            const double frac = (dv > 0.0) ? (V_TH - x_pre[0]) / dv : 1.0;
            const double u_minus = x_pre[1] + frac * (x[1] - x_pre[1]);
            const double u_plus  = u_minus + d;
            const double f_v_minus = 0.04 * V_TH * V_TH + 5.0 * V_TH + 140.0 - u_minus + I_EXT;
            const double f_v_plus  = 0.04 * C_RESET * C_RESET + 5.0 * C_RESET + 140.0 - u_plus + I_EXT;
            const double f_u_minus = A_PARAM * (B_PARAM * V_TH - u_minus);
            const double f_u_plus  = A_PARAM * (B_PARAM * C_RESET - u_plus);
            const double denom = f_v_minus;
            const double s00 = f_v_plus / denom;
            const double s10 = (f_u_plus - f_u_minus) / denom;
            // apply reset to state and saltation to tangent
            x[0] = C_RESET; x[1] = u_plus;
            const double w0 = x[2], w1 = x[3];
            x[2] = s00 * w0;             // S row0: [s00, 0]
            x[3] = s10 * w0 + w1;        // S row1: [s10, 1]
            // renormalize, accumulate growth once past transient
            const double n = std::hypot(x[2], x[3]);
            if (n > 0) { x[2] /= n; x[3] /= n; }
            if (t >= t_transient) {
                if (t_acc_start < 0) t_acc_start = t;
                log_sum += std::log(n);
            }
        }
    }
    const double T = (t_acc_start > 0) ? (t - t_acc_start) : t_run;
    return log_sum / T;
}

// ---- semi-analytic Poincare return map P and its multiplier -----------------
// P(u_cross): given u recorded at a v=V_TH crossing, apply reset (v<-C_RESET,
// u<-u_cross+d), integrate the flow to the NEXT upward v=V_TH crossing, return
// the interpolated u there. This is the first-return map on the section. The
// period-1 fixed point u* solves P(u*)=u*; its multiplier m=P'(u*) gives the
// second-iterate slope C=m^2, eps_local=C-1. Smooth in d by construction (no
// cloud fitting, immune to the periodic-window / band contamination).
static double P_map(double d, double u_cross_in, double dt_max, double t_cap) {
    auto stepper = odeint::make_controlled(ATOL, RTOL,
                     odeint::runge_kutta_cash_karp54<state_type>());
    RHS rhs{d};
    state_type x{C_RESET, u_cross_in + d};   // post-reset state
    double t = 0.0;
    while (t < t_cap) {
        state_type xpre = x; double dt = dt_max;
        odeint::controlled_step_result res;
        do { res = stepper.try_step(rhs, x, t, dt); } while (res == odeint::fail);
        if (xpre[0] < V_TH && x[0] >= V_TH) {
            const double dv = x[0] - xpre[0];
            const double frac = (dv > 0.0) ? (V_TH - xpre[0]) / dv : 1.0;
            return xpre[1] + frac * (x[1] - xpre[1]);   // u at next crossing
        }
    }
    return std::nan("");
}

// multiplier P'(u) by central finite difference (one ISI, no chaos build-up)
static double P_prime(double d, double u, double dt_max, double t_cap, double h = 1e-5) {
    double pp = P_map(d, u + h, dt_max, t_cap);
    double pm = P_map(d, u - h, dt_max, t_cap);
    return (pp - pm) / (2.0 * h);
}

// Newton solve for the period-1 fixed point u* of P near a seed, return
// {u_star, m=P'(u*), C=m^2, eps=C-1}. NaNs on failure.
struct MapDeriv { double u_star, m, C, eps; bool ok; };
static MapDeriv map_deriv(double d, double u_seed, double dt_max, double t_cap) {
    double u = u_seed;
    for (int it = 0; it < 60; ++it) {
        double pu = P_map(d, u, dt_max, t_cap);
        if (std::isnan(pu)) return {0,0,0,0,false};
        double g = pu - u;
        if (std::fabs(g) < 1e-11) break;
        double mp = P_prime(d, u, dt_max, t_cap);
        double gp = mp - 1.0;
        if (std::fabs(gp) < 1e-12) return {0,0,0,0,false};
        double step = g / gp;
        // damp large steps for stability
        if (std::fabs(step) > 5.0) step = (step > 0 ? 5.0 : -5.0);
        u -= step;
        if (u < -130 || u > -70) return {0,0,0,0,false};
    }
    double m = P_prime(d, u, dt_max, t_cap);
    double C = m * m;              // second-iterate slope at the fixed point
    return {u, m, C, C - 1.0, true};
}

// ---- INDEPENDENT cross-check: two-trajectory Benettin at the section --------
// Integrate a 2D (v,u) trajectory to its next upward v=V_TH crossing, apply
// reset, return u at the crossing (interpolated). Advances (x,t) in place.
template <class Stepper>
static double advance_to_spike(Stepper &stepper,
        RHS &rhs, state_type &x, double &t, double d, double dt_max, double t_max) {
    while (t < t_max) {
        state_type xpre = x; double dt = dt_max;
        odeint::controlled_step_result res;
        do { res = stepper.try_step(rhs, x, t, dt); } while (res == odeint::fail);
        if (xpre[0] < V_TH && x[0] >= V_TH) {
            const double dv = x[0] - xpre[0];
            const double frac = (dv > 0.0) ? (V_TH - xpre[0]) / dv : 1.0;
            const double u_cross = xpre[1] + frac * (x[1] - xpre[1]);
            x[0] = C_RESET; x[1] = u_cross + d;
            return u_cross;
        }
    }
    return std::nan("");
}

// Benettin largest exponent: reference + perturbed (u only) trajectories,
// renormalize the u-separation to delta0 at each spike, accumulate ln growth.
static double lyapunov_benettin(double d, double v0, double u0,
                                double t_transient, double t_run, double dt_max) {
    const double delta0 = 1e-7;
    auto s_ref = odeint::make_controlled(ATOL, RTOL, odeint::runge_kutta_cash_karp54<state_type>());
    auto s_prt = odeint::make_controlled(ATOL, RTOL, odeint::runge_kutta_cash_karp54<state_type>());
    RHS rhs{d};
    state_type xr{v0, u0}, xp{v0, u0 + delta0};
    double tr = 0.0, tp = 0.0;
    const double t_end = t_transient + t_run;
    double log_sum = 0.0, t_acc0 = -1.0, t_last = 0.0;
    while (tr < t_end) {
        double ur = advance_to_spike(s_ref, rhs, xr, tr, d, dt_max, t_end);
        // drive perturbed to the SAME spike index (single spike step)
        double up = advance_to_spike(s_prt, rhs, xp, tp, d, dt_max, t_end);
        if (std::isnan(ur) || std::isnan(up)) break;
        const double dsep = xp[1] - xr[1];       // separation in u post-reset
        const double sep = std::fabs(dsep);
        if (sep > 0 && tr >= t_transient) {
            if (t_acc0 < 0) { t_acc0 = tr; }
            log_sum += std::log(sep / delta0);
            t_last = tr;
        }
        // renormalize perturbed back to delta0 along the same sign, keep tp=tr
        if (sep > 0) { xp[1] = xr[1] + (dsep / sep) * delta0; xp[0] = xr[0]; tp = tr; }
    }
    const double T = (t_acc0 > 0) ? (t_last - t_acc0) : t_run;
    return (T > 0) ? log_sum / T : std::nan("");
}

// ---- laminar detection on a crossing sequence -------------------------------
struct Episodes { std::vector<double> reinj; std::vector<double> reinj2; std::vector<int> l_iter; };

static Episodes detect_laminar(const std::vector<double> &cross, double fp,
                               double clam, int skip_spikes) {
    Episodes ep;
    if ((int)cross.size() < skip_spikes + 4) return ep;
    bool in_lam = false;
    bool need_r2 = false;
    int start_k = 0;
    double reinj_val = 0.0, reinj2_val = 0.0;
    for (int k = skip_spikes + 2; k < (int)cross.size(); ++k) {
        const double ub  = cross[k]     - fp;
        const double ub2 = cross[k - 2] - fp;
        if (need_r2) { reinj2_val = ub; need_r2 = false; }
        if (std::fabs(ub) <= clam && std::fabs(ub2) > clam) {
            if (!in_lam) { start_k = k; reinj_val = ub; need_r2 = true; in_lam = true; }
        } else if (std::fabs(ub2) <= clam && std::fabs(ub) > clam) {
            // only exit on same branch as entry (same parity)
            if (in_lam && ((k - start_k) % 2 == 0)) {
                int ll = (k - start_k) / 2;
                if (ll >= 1) { ep.reinj.push_back(reinj_val); ep.reinj2.push_back(reinj2_val); ep.l_iter.push_back(ll); }
                in_lam = false;
            }
        }
    }
    return ep;
}

// ---- collect mode: one trajectory, ONLINE laminar detection + watchdog ------
// Detects episodes as crossings arrive (2-step look-back, identical logic to
// detect_laminar). Two safeguards:
//   * idle watchdog: if the trajectory spends > t_idle (ms) NOT in a laminar
//     phase and without entering one, abandon it (unproductive IC). It never
//     fires while a laminar phase is in progress -> the long-l tail is safe.
//   * l_safety cap: a single laminar phase exceeding l_safety F^2-iterates is a
//     stuck periodic orbit, not intermittency -> censored, trajectory abandoned.
struct CollectOut { Episodes ep; long censored = 0; double t_ran = 0.0; };

static CollectOut run_trajectory_collect(double d, double v0, double u0, double fp,
        double clam, int skip_spikes, double t_transient, double t_run_cap,
        double dt_max, double t_idle, int l_safety) {
    CollectOut out;
    auto stepper = odeint::make_controlled(ATOL, RTOL,
                     odeint::runge_kutta_cash_karp54<state_type>());
    RHS rhs{d};
    state_type x{v0, u0};
    double t = 0.0;
    const double t_end = t_transient + t_run_cap;
    int k = -1;                    // crossing index (post-transient)
    double prev1 = 0, prev2 = 0;   // cross[k-1], cross[k-2]
    bool in_lam = false;
    bool need_r2 = false;
    int start_k = 0; double reinj_val = 0.0, reinj2_val = 0.0;
    double t_last_activity = t_transient;   // time of last laminar start / transient end

    while (t < t_end) {
        state_type xpre = x; double dt = dt_max;
        odeint::controlled_step_result res;
        do { res = stepper.try_step(rhs, x, t, dt); } while (res == odeint::fail);
        if (!(xpre[0] < V_TH && x[0] >= V_TH)) continue;
        const double dv = x[0] - xpre[0];
        const double frac = (dv > 0.0) ? (V_TH - xpre[0]) / dv : 1.0;
        const double u_cross = xpre[1] + frac * (x[1] - xpre[1]);
        x[0] = C_RESET; x[1] = u_cross + d;        // reset from interpolated crossing
        if (t < t_transient) continue;
        ++k;
        if (k >= skip_spikes + 2) {
            const double ub  = u_cross - fp;
            const double ub2 = prev2 - fp;
            if (need_r2) { reinj2_val = ub; need_r2 = false; }
            if (std::fabs(ub) <= clam && std::fabs(ub2) > clam) {
                if (!in_lam) { start_k = k; reinj_val = ub; need_r2 = true; in_lam = true;
                               t_last_activity = t; }
            } else if (std::fabs(ub2) <= clam && std::fabs(ub) > clam) {
                // only exit on same branch as entry (same parity)
                if (in_lam && ((k - start_k) % 2 == 0)) {
                    int ll = (k - start_k) / 2;
                    if (ll >= 1) { out.ep.reinj.push_back(reinj_val);
                                   out.ep.reinj2.push_back(reinj2_val);
                                   out.ep.l_iter.push_back(ll); }
                    in_lam = false; t_last_activity = t;
                }
            }
            if (in_lam && (k - start_k) / 2 > l_safety) { ++out.censored; break; }
            if (!in_lam && (t - t_last_activity) > t_idle) break;
        }
        prev2 = prev1; prev1 = u_cross;
    }
    out.t_ran = t;
    return out;
}

// ---- escape mode: deterministic laminar length l(u0) from a chosen entry ----
// Launch from the post-reset state of a reinjection at ubar=u0: (C_RESET, u0+fp+d).
// Integrate, build the crossing sequence with the entry prepended, and read off
// the laminar length by the SAME 2-step convention as detect_laminar
// (l_spikes = k_escape - 0 ; l_iter = (l_spikes+1)/2). Deterministic in u0 (the
// section fixes v), so ONE launch per u0 gives l(u0). Returns l_safety+1 if the
// trajectory never escapes within the cap (stuck periodic / at fixed point).
static int escape_length(double d, double u0, double fp, double clam,
                         double dt_max, int l_safety, double t_cap) {
    auto stepper = odeint::make_controlled(ATOL, RTOL,
                     odeint::runge_kutta_cash_karp54<state_type>());
    RHS rhs{d};
    state_type x{C_RESET, u0 + fp + d};   // post-reset state of the entry crossing
    double t = 0.0;
    std::vector<double> ub;               // ubar at each crossing; ub[0]=entry
    ub.reserve(2 * l_safety + 8);
    ub.push_back(u0);
    while (t < t_cap) {
        state_type xpre = x; double dt = dt_max;
        odeint::controlled_step_result res;
        do { res = stepper.try_step(rhs, x, t, dt); } while (res == odeint::fail);
        if (!(xpre[0] < V_TH && x[0] >= V_TH)) continue;
        const double dv = x[0] - xpre[0];
        const double frac = (dv > 0.0) ? (V_TH - xpre[0]) / dv : 1.0;
        const double u_cross = xpre[1] + frac * (x[1] - xpre[1]);
        x[0] = C_RESET; x[1] = u_cross + d;        // reset from interpolated crossing
        ub.push_back(u_cross - fp);
        int k = (int)ub.size() - 1;
        if (k >= 2 && k % 2 == 0) {       // same branch as entry (k=0)
            if (std::fabs(ub[k - 2]) <= clam && std::fabs(ub[k]) > clam)
                return k / 2;                        // escaped; l_iter
            if (k / 2 > l_safety) return l_safety + 1;   // stuck
        }
    }
    return l_safety + 1;
}

// ---------------------------------------------------------------------------
struct Cfg {
    std::string mode = "episodes";
    double d = -11.83;
    int trials = 1000;
    uint64_t seed = 12345;
    double t_transient = 2000.0;
    double t_run = 100000.0;
    double dt_max = 0.5;
    int skip_spikes = 3;
    double fp = FP_SHIFT;
    std::vector<double> clam = {2.0, 1.65};
    double map_w = 0.6;          // mapcurve grid half-width
    int map_n = 81;              // mapcurve grid points
    // collect mode
    long collect_until = 1000000;   // stop when this many episodes gathered
    double t_idle = 20000.0;        // ms outside laminar before abandoning IC
    int l_safety = 100000;          // laminar length (F^2) above = stuck periodic
    double max_work = 5e9;          // global integration-time budget (ms)
    // escape mode: uniform u0 grid
    int u0_n = 20000;               // number of entry points
    double u0_lo = 0.0, u0_hi = 0.0; // defaults to [-clam,clam] when both 0
    std::string out_prefix = "out/run";
    // IC sampling box
    double v_lo = -70, v_hi = -50, u_lo = -110, u_hi = -90;
};

static std::vector<double> parse_list(const std::string &s) {
    std::vector<double> v; std::stringstream ss(s); std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(std::stod(tok));
    return v;
}

int main(int argc, char **argv) {
    Cfg c;
    for (int i = 1; i < argc; ++i) {
        std::string o = argv[i];
        auto val = [&]() { return std::string(argv[++i]); };
        if      (o == "--mode") c.mode = val();
        else if (o == "--d") c.d = std::stod(val());
        else if (o == "--trials") c.trials = std::stoi(val());
        else if (o == "--seed") c.seed = std::stoull(val());
        else if (o == "--t_transient") c.t_transient = std::stod(val());
        else if (o == "--t_run") c.t_run = std::stod(val());
        else if (o == "--dt_max") c.dt_max = std::stod(val());
        else if (o == "--skip_spikes") c.skip_spikes = std::stoi(val());
        else if (o == "--fp") c.fp = std::stod(val());
        else if (o == "--clam") c.clam = parse_list(val());
        else if (o == "--map_w") c.map_w = std::stod(val());
        else if (o == "--map_n") c.map_n = std::stoi(val());
        else if (o == "--collect_until") c.collect_until = std::stoll(val());
        else if (o == "--t_idle") c.t_idle = std::stod(val());
        else if (o == "--l_safety") c.l_safety = std::stoi(val());
        else if (o == "--max_work") c.max_work = std::stod(val());
        else if (o == "--u0_n") c.u0_n = std::stoi(val());
        else if (o == "--u0_lo") c.u0_lo = std::stod(val());
        else if (o == "--u0_hi") c.u0_hi = std::stod(val());
        else if (o == "--out_prefix") c.out_prefix = val();
        else { std::cerr << "Unknown option: " << o << "\n"; return 1; }
    }

    if (c.mode == "mapdump") {
        // one long trajectory; dump (u_n, u_{n+2}) pairs after transient
        std::mt19937_64 rng(c.seed);
        std::uniform_real_distribution<double> uv(c.v_lo, c.v_hi), uu(c.u_lo, c.u_hi);
        auto cr = crossing_sequence(c.d, uv(rng), uu(rng), c.t_transient, c.t_run, c.dt_max);
        std::ofstream f(c.out_prefix + "_map.dat");
        f << std::setprecision(15);
        f << "# d=" << c.d << "  n_cross=" << cr.size() << "\n# u_n  u_np2\n";
        for (size_t k = 0; k + 2 < cr.size(); ++k) f << cr[k] << " " << cr[k + 2] << "\n";
        std::cerr << "mapdump d=" << c.d << " crossings=" << cr.size() << "\n";
        return 0;
    }

    if (c.mode == "lyap") {
        // ensemble-averaged largest Lyapunov exponent over `trials` ICs
        std::vector<double> lam(c.trials);
        #pragma omp parallel for schedule(dynamic)
        for (int tr = 0; tr < c.trials; ++tr) {
            std::mt19937_64 rng(c.seed + (uint64_t)tr * 0x9E3779B97F4A7C15ULL);
            std::uniform_real_distribution<double> uv(c.v_lo, c.v_hi), uu(c.u_lo, c.u_hi);
            lam[tr] = lyapunov_max(c.d, uv(rng), uu(rng), c.t_transient, c.t_run, c.dt_max);
        }
        double s = 0, s2 = 0;
        for (double l : lam) { s += l; s2 += l * l; }
        double mean = s / c.trials;
        double sd = std::sqrt(std::max(0.0, s2 / c.trials - mean * mean));
        double se = sd / std::sqrt((double)c.trials);
        // machine-readable single line to stdout: d lambda1 se
        std::cout << std::setprecision(12) << c.d << " " << mean << " " << se << "\n";
        std::cerr << "lyap d=" << c.d << " lambda1=" << mean << " +/- " << se
                  << " (" << c.trials << " ICs)\n";
        return 0;
    }

    if (c.mode == "escape") {
        // deterministic l(u0) on a uniform u0 grid (stratified: dense in center)
        const double clam = c.clam.empty() ? 1.65 : c.clam[0];
        double lo = c.u0_lo, hi = c.u0_hi;
        if (lo == 0.0 && hi == 0.0) { lo = -clam; hi = clam; }
        std::vector<double> u0s(c.u0_n), lval(c.u0_n);
        for (int i = 0; i < c.u0_n; ++i)
            u0s[i] = lo + (hi - lo) * (i + 0.5) / c.u0_n;
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < c.u0_n; ++i) {
            if (std::fabs(u0s[i]) < 1e-4) { lval[i] = c.l_safety + 1; continue; }
            lval[i] = escape_length(c.d, u0s[i], c.fp, clam, c.dt_max,
                                    c.l_safety, c.t_run);
        }
        std::ofstream f(c.out_prefix + "_escape.dat");
        f << std::setprecision(12) << "# u0  l(u0)  (deterministic escape length)\n";
        long nstuck = 0;
        for (int i = 0; i < c.u0_n; ++i) {
            f << u0s[i] << " " << lval[i] << "\n";
            if (lval[i] > c.l_safety) ++nstuck;
        }
        std::cerr << "escape d=" << c.d << " grid=" << c.u0_n << " on ["
                  << lo << "," << hi << "] stuck=" << nstuck << "\n";
        return 0;
    }

    if (c.mode == "collect") {
        // collect-until-N episodes at ONE clam, u*-centered by --fp, with the
        // idle watchdog + laminar safety cap. Global work budget gates periodic d.
        const double clam = c.clam.empty() ? 1.65 : c.clam[0];
        const double t_run_cap = c.t_run;   // per-trajectory time cap
        std::atomic<long> got{0}, work_ms{0}, ntraj{0}, censored{0};
        std::vector<Episodes> pool;
        #pragma omp parallel
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            std::mt19937_64 rng(c.seed + (uint64_t)(tid + 1) * 0x9E3779B97F4A7C15ULL);
            std::uniform_real_distribution<double> uv(c.v_lo, c.v_hi), uu(c.u_lo, c.u_hi);
            Episodes local;
            while (got.load(std::memory_order_relaxed) < c.collect_until &&
                   (double)work_ms.load(std::memory_order_relaxed) < c.max_work) {
                auto r = run_trajectory_collect(c.d, uv(rng), uu(rng), c.fp, clam,
                             c.skip_spikes, c.t_transient, t_run_cap, c.dt_max,
                             c.t_idle, c.l_safety);
                local.reinj.insert(local.reinj.end(), r.ep.reinj.begin(), r.ep.reinj.end());
                local.reinj2.insert(local.reinj2.end(), r.ep.reinj2.begin(), r.ep.reinj2.end());
                local.l_iter.insert(local.l_iter.end(), r.ep.l_iter.begin(), r.ep.l_iter.end());
                got.fetch_add((long)r.ep.l_iter.size(), std::memory_order_relaxed);
                work_ms.fetch_add((long)r.t_ran, std::memory_order_relaxed);
                ntraj.fetch_add(1, std::memory_order_relaxed);
                censored.fetch_add(r.censored, std::memory_order_relaxed);
            }
            #pragma omp critical
            {
                pool.push_back(std::move(local));
            }
        }
        // merge
        Episodes all;
        for (auto &p : pool) {
            all.reinj.insert(all.reinj.end(), p.reinj.begin(), p.reinj.end());
            all.reinj2.insert(all.reinj2.end(), p.reinj2.begin(), p.reinj2.end());
            all.l_iter.insert(all.l_iter.end(), p.l_iter.begin(), p.l_iter.end());
        }
        std::ostringstream tag; tag << std::setprecision(4) << clam;
        std::ofstream f(c.out_prefix + "_ep_clam" + tag.str() + ".dat");
        f << std::setprecision(12) << "# reinj  reinj2  l_iter\n";
        double s = 0, s2 = 0; int mx = 0;
        for (size_t i = 0; i < all.l_iter.size(); ++i) {
            f << all.reinj[i] << " " << all.reinj2[i] << " " << all.l_iter[i] << "\n";
            double li = all.l_iter[i]; s += li; s2 += li * li; mx = std::max(mx, all.l_iter[i]);
        }
        long N = (long)all.l_iter.size();
        double mean = N ? s / N : 0.0;
        double var = N ? std::max(0.0, s2 / N - mean * mean) : 0.0;
        bool reached = N >= c.collect_until;
        std::ofstream sum(c.out_prefix + "_summary.csv");
        sum << std::setprecision(12)
            << "d,clam,mean_l,max_l,std_l,N,n_traj,censored,work_ms,reached_target\n"
            << c.d << "," << clam << "," << mean << "," << mx << "," << std::sqrt(var)
            << "," << N << "," << ntraj.load() << "," << censored.load() << ","
            << work_ms.load() << "," << (reached ? 1 : 0) << "\n";
        std::cerr << "collect d=" << c.d << " N=" << N << "/" << c.collect_until
                  << (reached ? " [reached]" : " [budget-limited: low intermittency]")
                  << " mean_l=" << mean << " max_l=" << mx
                  << " traj=" << ntraj.load() << " censored=" << censored.load()
                  << " work=" << (work_ms.load() / 1e6) << "e6 ms\n";
        return 0;
    }

    if (c.mode == "mapderiv") {
        // semi-analytic C(d)=P'(u*)^2 via Newton on the return map
        const double t_cap = 5000.0;
        MapDeriv r = map_deriv(c.d, c.fp, c.dt_max, t_cap);
        if (!r.ok) { std::cout << c.d << " nan nan nan nan\n";
                     std::cerr << "mapderiv d=" << c.d << " FAILED\n"; return 0; }
        std::cout << std::setprecision(12) << c.d << " " << r.u_star << " "
                  << r.m << " " << r.C << " " << r.eps << "\n";
        std::cerr << "mapderiv d=" << c.d << " u*=" << r.u_star << " m=" << r.m
                  << " C=" << r.C << " eps_local=" << r.eps << "\n";
        return 0;
    }

    if (c.mode == "mapcurve") {
        // deterministic second-return map P^2 on a grid around the fixed point
        // u*, for a clean cubic fit  F2(u')-u* = C*s + A*s^3  (s = u'-u*).
        // Python extracts A (drift cubic) and eps=C-1. No cloud contamination.
        const double t_cap = 5000.0;
        MapDeriv r = map_deriv(c.d, c.fp, c.dt_max, t_cap);
        if (!r.ok) { std::cerr << "mapcurve d=" << c.d << " FAILED (no u*)\n";
                     std::cout << "# FAILED\n"; return 0; }
        const double us = r.u_star, w = c.map_w;
        const int ng = c.map_n;
        std::cout << std::setprecision(12) << "# d=" << c.d << " u_star=" << us
                  << " m=" << r.m << " C=" << r.C << "\n# s  F2_minus_ustar\n";
        for (int i = 0; i < ng; ++i) {
            double s = -w + 2 * w * i / (ng - 1);
            double p1 = P_map(c.d, us + s, c.dt_max, t_cap);
            if (std::isnan(p1)) continue;
            double p2 = P_map(c.d, p1, c.dt_max, t_cap);
            if (std::isnan(p2)) continue;
            std::cout << s << " " << (p2 - us) << "\n";
        }
        return 0;
    }

    if (c.mode == "lyapfd") {
        // independent Benettin cross-check of lambda1
        std::vector<double> lam(c.trials);
        #pragma omp parallel for schedule(dynamic)
        for (int tr = 0; tr < c.trials; ++tr) {
            std::mt19937_64 rng(c.seed + (uint64_t)tr * 0x9E3779B97F4A7C15ULL);
            std::uniform_real_distribution<double> uv(c.v_lo, c.v_hi), uu(c.u_lo, c.u_hi);
            lam[tr] = lyapunov_benettin(c.d, uv(rng), uu(rng), c.t_transient, c.t_run, c.dt_max);
        }
        double s = 0, s2 = 0; int nv = 0;
        for (double l : lam) if (!std::isnan(l)) { s += l; s2 += l * l; ++nv; }
        double mean = nv ? s / nv : std::nan("");
        double sd = nv ? std::sqrt(std::max(0.0, s2 / nv - mean * mean)) : 0.0;
        std::cout << std::setprecision(12) << c.d << " " << mean << " "
                  << (nv ? sd / std::sqrt((double)nv) : 0.0) << "\n";
        std::cerr << "lyapfd d=" << c.d << " lambda1=" << mean << " (" << nv << " ICs)\n";
        return 0;
    }

    if (c.mode != "episodes") { std::cerr << "bad --mode\n"; return 1; }

    // episodes: parallel trials, detect at each clam, pool episodes
    const int nclam = (int)c.clam.size();
    std::vector<Episodes> pooled(nclam);
    #pragma omp parallel
    {
        std::vector<Episodes> local(nclam);
        #pragma omp for schedule(dynamic)
        for (int tr = 0; tr < c.trials; ++tr) {
            std::mt19937_64 rng(c.seed + (uint64_t)tr * 0x9E3779B97F4A7C15ULL);
            std::uniform_real_distribution<double> uv(c.v_lo, c.v_hi), uu(c.u_lo, c.u_hi);
            auto cr = crossing_sequence(c.d, uv(rng), uu(rng), c.t_transient, c.t_run, c.dt_max);
            for (int ci = 0; ci < nclam; ++ci) {
                auto ep = detect_laminar(cr, c.fp, c.clam[ci], c.skip_spikes);
                auto &L = local[ci];
                L.reinj.insert(L.reinj.end(), ep.reinj.begin(), ep.reinj.end());
                L.reinj2.insert(L.reinj2.end(), ep.reinj2.begin(), ep.reinj2.end());
                L.l_iter.insert(L.l_iter.end(), ep.l_iter.begin(), ep.l_iter.end());
            }
        }
        #pragma omp critical
        for (int ci = 0; ci < nclam; ++ci) {
            pooled[ci].reinj.insert(pooled[ci].reinj.end(), local[ci].reinj.begin(), local[ci].reinj.end());
            pooled[ci].reinj2.insert(pooled[ci].reinj2.end(), local[ci].reinj2.begin(), local[ci].reinj2.end());
            pooled[ci].l_iter.insert(pooled[ci].l_iter.end(), local[ci].l_iter.begin(), local[ci].l_iter.end());
        }
    }

    // write per-clam episode files + summary
    std::ofstream sum(c.out_prefix + "_summary.csv");
    sum << std::setprecision(12);
    sum << "d,seed,clam,mean_l,max_l,std_l,N\n";
    for (int ci = 0; ci < nclam; ++ci) {
        const auto &L = pooled[ci];
        std::ostringstream tag; tag << std::setprecision(4) << c.clam[ci];
        std::ofstream f(c.out_prefix + "_ep_clam" + tag.str() + ".dat");
        f << std::setprecision(12) << "# reinj  reinj2  l_iter\n";
        double s = 0, s2 = 0; int mx = 0;
        for (size_t i = 0; i < L.l_iter.size(); ++i) {
            f << L.reinj[i] << " " << L.reinj2[i] << " " << L.l_iter[i] << "\n";
            double li = L.l_iter[i]; s += li; s2 += li * li; mx = std::max(mx, L.l_iter[i]);
        }
        size_t N = L.l_iter.size();
        double mean = N ? s / N : 0.0;
        double var = N ? std::max(0.0, s2 / N - mean * mean) : 0.0;
        sum << c.d << "," << c.seed << "," << c.clam[ci] << "," << mean << ","
            << mx << "," << std::sqrt(var) << "," << N << "\n";
        std::cerr << "clam=" << c.clam[ci] << " N=" << N << " mean_l=" << mean
                  << " max_l=" << mx << "\n";
    }
    return 0;
}
