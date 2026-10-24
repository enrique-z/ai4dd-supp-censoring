#!/bin/bash
set -e
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
S="python3.14 -E -P s4_optimizer_sensitivity.py"
date
for L in 1e-3 1e-1 1e-2 1; do $S sweep --lam $L --nproc 12 > sweep_lam_$L.log 2>&1; echo "done lam=$L $(date)"; done
$S sweep --orig --lam 1e-3 --nproc 12 > sweep_orig_full.log 2>&1; echo "done orig_full $(date)"
$S summarise > summarise.log 2>&1; echo "ALLDONE $(date)"
