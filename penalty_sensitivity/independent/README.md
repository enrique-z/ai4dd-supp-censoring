# Independent re-implementation of the S4 penalty analysis

Written separately from `../s4_optimizer_sensitivity.py`, which these scripts do not import or read. They reuse only the database, the target list and the random seeds of the original split and size-matched subsample.

| script | what it does |
|---|---|
| `extract.py [nproc]` | Runs the per-target SQL and featurisation, and writes `cache/` |
| `ridge_cv.py [n_seeds nproc tids...]` | Fits naive, exact-only and size-matched arms as plain ridge (eigendecomposition, exact for every alpha). Alpha comes from 5-fold CV on each arm's training rows, from the penalty implied by the original objective, or from a fixed grid. Writes `ridge_results_*.json` |
| `analyze.py <results.json>` | Computes contrasts, bootstrap CIs (10,000 resamples over targets) and ρ, and writes `summary.json` |
| `sweep_orig_param.py` | Refits all four arms, Tobit included, on the original objective at λ ∈ {1e-3, 1e-2, 1e-1, 1}. Writes `sweep_results_s10_n158.json` |
| `analyze_sweep.py <sweep.json>` | Writes `sweep_summary.json` |

**Results.**
- The CV-tuned naive − exact-only contrast is +0.0215 [0.0188, 0.0243], with naive worse on 152 of 158 targets (`summary.json`).
- The fixed-λ sweep agrees with `../s4_sensitivity_summary.json` to within 0.002 in every contrast.
