#!/bin/bash
# multiseed_ci.sh — Run episodes with N independent seeds, collect per-seed
# mean laminar lengths, compute true SE from independent experiments.
#
# Usage: bash multiseed_ci.sh [-d D_VALUE] [-n NSEEDS] [-t TRIALS_PER_SEED]
#
# Defaults: d=-11.90, 20 seeds, 100 trials/seed, clam=2.0

set -euo pipefail

BIN=/home/juan/Systems_Biology_Izhikevich/cpp/izh_intermittency
OUTDIR=/home/juan/Systems_Biology_Izhikevich/cpp/out/multiseed
CLAM=2.0

D="-11.90"
NSEEDS=20
TRIALS=100

while getopts "d:n:t:" opt; do
    case $opt in
        d) D="$OPTARG" ;;
        n) NSEEDS="$OPTARG" ;;
        t) TRIALS="$OPTARG" ;;
        *) echo "Usage: $0 [-d D] [-n NSEEDS] [-t TRIALS]"; exit 1 ;;
    esac
done

TAG=$(echo "$D" | tr -d '-' | tr '.' 'p')
RUNDIR="$OUTDIR/d${TAG}"
mkdir -p "$RUNDIR"

echo "=== multiseed: d=$D, $NSEEDS seeds × $TRIALS trials, clam=$CLAM ==="

# Run all seeds in parallel (GNU parallel or sequential fallback)
run_seed() {
    local seed=$1
    local prefix="$RUNDIR/seed${seed}"
    $BIN --mode episodes --d "$D" --trials "$TRIALS" --seed "$seed" \
         --clam "$CLAM" --t_transient 2000 --t_run 100000 \
         --out_prefix "$prefix" 2>/dev/null
}
export -f run_seed
export BIN D TRIALS CLAM RUNDIR

if command -v parallel &>/dev/null; then
    seq 1 "$NSEEDS" | parallel -j "$(nproc)" run_seed {}
else
    for s in $(seq 1 "$NSEEDS"); do
        run_seed "$s" &
    done
    wait
fi

# Collect per-seed means from summary CSVs
RESULT="$RUNDIR/multiseed_summary.csv"
echo "seed,d,clam,mean_l,max_l,std_l,N" > "$RESULT"

for s in $(seq 1 "$NSEEDS"); do
    csv="$RUNDIR/seed${s}_summary.csv"
    if [[ -f "$csv" ]]; then
        tail -1 "$csv" | awk -F',' -v s="$s" '{print s","$0}' >> "$RESULT"
    fi
done

# Compute overall mean, SE, 95% CI from independent seed means
python3 - "$RESULT" <<'PYEOF'
import sys, numpy as np
f = sys.argv[1]
data = np.genfromtxt(f, delimiter=',', skip_header=1)
if data.ndim == 1:
    data = data.reshape(1, -1)
means = data[:, 4]  # mean_l column (index 4: seed,d,seed_in_csv,clam,mean_l,...)
Ns = data[:, 7]     # N column
n = len(means)
grand_mean = np.mean(means)
se = np.std(means, ddof=1) / np.sqrt(n)
ci95 = 1.96 * se
total_episodes = int(np.sum(Ns))
print(f"\n{'='*55}")
print(f"  d = {data[0,2]:.2f}   clam = {data[0,3]}")
print(f"  {n} independent seeds × {int(Ns[0])} episodes (approx)")
print(f"  Total episodes: {total_episodes}")
print(f"  Per-seed means: {means}")
print(f"  Grand mean <l>  = {grand_mean:.4f}")
print(f"  SE (from seeds) = {se:.4f}")
print(f"  95% CI          = [{grand_mean-ci95:.4f}, {grand_mean+ci95:.4f}]")
print(f"{'='*55}\n")
PYEOF

echo "Results in $RESULT"
