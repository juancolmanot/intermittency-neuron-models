# Intermittency in neuron models — reproducibility package

Code and data for *Reinjection processes in intermittent neuron models: the Rulkov map and the Izhikevich model* (Colman & Elaskar, Frontiers, 2026, under review).

Intermittency analysis of the Izhikevich and Rulkov neuron models using the
M-function methodology.

Juan Colman and Sergio Elaskar (2026)

---

## Directory Structure

```
supplementary/
├── README.md              ← this file
├── src/                   ← C++ simulation source code
├── scripts/               ← Python analysis & figure-generation scripts
├── data/                  ← raw numerical data (simulations + theory)
└── multiseed/             ← independent-experiment standard errors
```

---

## src/ — Simulation Source Code

| File | Description |
|------|-------------|
| `izh_intermittency.cpp` | Izhikevich neuron model: 8 modes (episodes, lyap, mapdump, escape, collect, mapderiv, mapcurve, lyapfd). OpenMP-parallelized. |
| `Makefile` | Build for izh_intermittency (g++ with OpenMP) |
| `rulkov_reinjection.cpp` | Rulkov 1D map: iterate near saddle-node, collect reinjection points |
| `rulkov_scaling.cpp` | Rulkov scaling sweep: mean/max laminar lengths vs epsilon |
| `rulkov_scaling_v2.cpp` | Rulkov scaling v2: improved sampling with polynomial coefficients |
| `Makefile_rulkov` | Build for Rulkov C++ programs |

### Building

```bash
# Izhikevich
cd src/
make            # produces izh_intermittency binary

# Rulkov
make -f Makefile_rulkov
```

### Running Izhikevich episodes

```bash
./izh_intermittency --mode episodes --d -11.90 --trials 1000 \
    --clam 2.0 --t_transient 50000 --t_run 100000 --out_prefix results
```

Output: `results_episodes.dat` (columns: reinj, reinj2, l_iter),
`results_summary.csv`, `results_meta.txt`.

---

## scripts/ — Analysis & Figure Generation

### Figure generators (manuscript figures)

| Script | Generates | Manuscript Figure |
|--------|-----------|-------------------|
| `fig_rulkov_evolution.py` | `rulkov_map_evolution.png` | Fig. 1 (Rulkov map evolution) |
| `fig_rulkov_bifurcation.py` | `Fig1.png` | Fig. 2 (Rulkov bifurcation diagram) |
| `rulkov_bifurcation_pub.py` | `rulkov_floquet_pub.png` | Fig. 3 (Floquet multiplier) |
| `rulkov_c001_regenerate.py` | `M_f_c=0.01.png`, `M1_f_c=0.01.png`, `M2_f_c=0.01.png`, `rpd_f_c=0.01.png`, `pdll_f_c=0.01.png`, `local_map_fit_c=0.01.png`, `rulkov_scaling_v2_c=0.01.png` | Figs. 8–13 (Rulkov c=0.01 analysis) |
| `rulkov_scaling_analysis.py` | `rulkov_scaling_v2_c=0.10.png` | Fig. 6 (Rulkov c=0.1 scaling) |
| `generate_figures.py` | `izhikevich_*.png` (7 figures) | Figs. 14–20 (Izhikevich model) |
| `pub_krause_d1190.py` | `pub_izh_M_d1190.png`, `pub_izh_branches_d1190.png`, `pub_izh_phi_d1190.png`, `pub_izh_l_d1190.png` | Figs. 21–24 (M-function analysis d=−11.90) |
| `fig_phi_map_derivative.py` | `phi_map_derivative_link.pdf` | Fig. 25 (φ–map derivative link) |
| `fractional_B_contribution.py` | `izhikevich_B_term_fraction.png` | Fig. 26 (B-term fraction) |
| `pub_single_d.py` | `pub_psi_m11p90.pdf` | Fig. 27 (ψ for d=−11.90) |
| `pub_izh_all_figures.py` | `pub_izh_scaling.pdf` | Fig. 28 (Izhikevich scaling law) |
| `report_laminar_saturation.py` | `fig2_laminar_surface.pdf`, `fig3_limit_and_rpd_gap.pdf` | Figs. 29–30 (saturation analysis) |

### Core analysis engine

| Script | Purpose |
|--------|---------|
| `M_rpd_pdll_krause.py` | M-function computation, branch fitting, RPD/PDLL calculation. Called by pub_krause_d1190.py and others. |
| `multiseed_ci.sh` | Independent-experiment standard errors (20 seeds × N trials) |

| `rulkov_c01_regenerate.py` | `rulkov_map_c=0.1.png`, `rulkov_local_map_c=0.1.png`, `rulkov_M_function_c=0.1.png`, `rulkov_rpd_function_c=0.1.png`, `rulkov_pdll_c=0.1.png` | Figs. 4–7 (Rulkov c=0.1 analysis) |

All manuscript figures have a generating script. No orphans.

---

## data/ — Raw Numerical Data

### Rulkov map data

| File | Contents |
|------|----------|
| `rulkov_scaling_v2_c=0.01.csv` | Scaling sweep c=0.01: log10(eps), epsilon, gamma, mean_l, max_l, std_l, N_episodes, polynomial coefficients |
| `rulkov_scaling_v2_c=0.10.csv` | Same for c=0.1 |
| `rulkov_reinjection_c=0.01.dat` | Reinjection points (shifted by fixed point) for c=0.01 |
| `rulkov_reinjection_c=0.10.dat` | Same for c=0.1 |
| `rulkov_characteristic_theoretical_c=0.01.csv` | Theoretical characteristic relation for c=0.01 |
| `rulkov_characteristic_theoretical_c=0.10.csv` | Same for c=0.1 |

### Izhikevich model data

| File | Contents |
|------|----------|
| `izhikevich_bifurcation_state_d_u_1.dat` | Bifurcation diagram: d vs u at Poincaré section |
| `lyapunov_bifurcation_izhikevich_d_1.dat` | Lyapunov exponent vs d |
| `poincare_fig19_d={-11,-12,-12.5,-13}.dat` | Poincaré return maps at selected d values |
| `izhikevich_poincare_map_v=30_d=-16.dat` | Wide return map at d=−16 |
| `d-{11.80,11.90,12.00,12.20}_ep_clam2.00.dat` | Episode data: columns reinj, reinj2, l_iter (shifted reinjection coordinates and laminar length in F² iterates) |
| `d-{11.80,11.90,12.00,12.20}_meta.txt` | Metadata: u_star (fixed point) and eps_local (local ε = C(d)−1) |
| `d-{11.80,11.90,12.00,12.20}_mapcurve.dat` | Numerical F² map curve near the tangency |
| `step3_theoretical_results.csv` | Theoretical predictions: d, eps, mean_l_sim, mean_l_th, ratio, RPD weights |

### Data format notes

- Episode files (`.dat`): 3 columns — reinj (reinjection u, shifted), reinj2 (2nd coordinate), l_iter (laminar length in F² steps)
- Scaling CSVs: header row describes columns; log10(eps) spans [−10, −3]
- Meta files: 2 numbers — u_star (F² fixed point), eps_local (C(d)−1)

---

## multiseed/ — Independent-Experiment Standard Errors

Each file contains results from 20 independent seeds (different RNG
initializations), each running N trials with fresh initial conditions.

| File | d value | Trials/seed |
|------|---------|-------------|
| `d11p80_summary.csv` | −11.80 | 100 |
| `d11p82_summary.csv` | −11.82 | 100 |
| `d11p90_summary.csv` | −11.90 | 100 |
| `d12p00_summary.csv` | −12.00 | 100 |
| `d12p20_summary.csv` | −12.20 | 50 |
| `d12p80_summary.csv` | −12.80 | 30 |

Columns: seed, d, clam, mean_l, max_l, std_l, N

Standard errors SE(⟨l⟩) < 0.06 across all d values, confirming numerical
stability of the laminar length statistics.

---

## Reproducing Key Results

### 1. Rulkov scaling law (Type-I, ε⁻¹/²)

```bash
python3 scripts/rulkov_scaling_analysis.py
# Reads data/rulkov_scaling_v2_c=0.10.csv
# Generates scaling plot with theory overlay
```

### 2. Izhikevich M-function analysis (d = −11.90)

```bash
# First generate episodes (requires compiled binary)
./src/izh_intermittency --mode episodes --d -11.90 --trials 1000 \
    --clam 2.0 --t_transient 50000 --t_run 100000 --out_prefix data/d-11.90

# Then run M-function analysis
python3 scripts/pub_krause_d1190.py
```

### 3. Izhikevich scaling law (Type-III, ε⁻¹)

```bash
python3 scripts/pub_izh_all_figures.py
# Reads episode data from data/d-*_ep_clam2.00.dat
```

### 4. Independent-experiment standard errors

```bash
bash scripts/multiseed_ci.sh -d -11.90 -n 20 -t 100
# Requires compiled izh_intermittency binary
```

---

## Software Requirements

- C++17 compiler (g++ ≥ 9) with OpenMP
- Python 3.8+ with: numpy, scipy, matplotlib
- GNU Make


---

## Revision additions (September 2026)

Added in response to the reviewers:

| File | Purpose |
|------|---------|
| `src/rulkov_cscan.cpp` | Reinjection collector for the Rulkov map parameterised in the laminar half-width `c` (exact `gamma_c`, `x*`). |
| `scripts/rulkov_c_scan.py` | Scan over `c` (Sec. 3.4.4, Fig. 7): Branch-1 fraction `P1(c)`, the two terms `q c^2` vs `eps`, RPD jump; compares with Eq. (27). Regenerates its own raw data (`data/rulkov_cscan/`, ~1 min) if absent. |
| `data/rulkov_cscan_summary.csv` | Output table of the scan. |
| `scripts/fig_rulkov_bifurcation.py` | Fig. 1 (top): bifurcation diagram of the fast map at `alpha_R = 4.8`, with `gamma_SN = -2.931` and the boundary crisis `gamma_EC = -4.263`. |
| `scripts/rulkov_c01_regenerate.py`, `scripts/rulkov_c001_regenerate.py` | Rulkov `c = 0.1` and `c = 0.01` pipelines (Secs. 3.3–3.4). Both use `eps = 3e-4`. |
| `scripts/report_laminar_saturation.py` | Figs. 12–13 (saturation), panel labels updated. |

Build the C++ tools with `g++ -O3 -o rulkov_cscan src/rulkov_cscan.cpp` (and see `src/Makefile*` for the rest). Python requirements: `numpy`, `scipy`, `matplotlib` (LaTeX text rendering is enabled in several scripts; install a TeX distribution or set `text.usetex = False`).

## License

Code: MIT License (see `LICENSE`). Data files under `data/` and `multiseed/`: CC BY 4.0.
