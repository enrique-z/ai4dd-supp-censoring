# S4 optimiser and regularisation sensitivity (post hoc)

This directory holds a post-hoc rerun of the S4 downstream benchmark from the ChEMBL censoring study. It has two parts:

- a numerically corrected Tobit fit
- a sweep of the ridge penalty, including a cross-validated (tuned) penalty

The original analysis code and evidence are not modified. The published per-seed results are read from `../evidence/censoring_raw.json`.

## Data and environment

- **Data:** ChEMBL 36 SQLite, opened read-only (`mode=ro&immutable=1`) from the path in the `CHEMBL_DB` environment variable (default: `./data/chembl_36/chembl_36_sqlite/chembl_36.db`).
- **Environment:** `python3.14 -E -P` with Python 3.14.7, numpy 2.3.4, scipy 1.17.1, RDKit 2026.03.1 and sqlite 3.54.0. Run with `PYTHONDONTWRITEBYTECODE=1`.

## Scripts

Run every script from this directory; paths are relative to it.

### `s4_optimizer_sensitivity.py`

This is a self-contained script. Every step except the fit function is copied from `analysis/run_censoring.py::s4_downstream`. That covers:

- the SQL, including the relation filter
- target selection
- per-compound aggregation
- Morgan fingerprints (r=2, 1024 bits)
- the exact-only test split (seeds 0–9, 25% test)
- the size-matched subsample

The script has these subcommands:

| subcommand | what it does |
|---|---|
| `extract [--shard i --nshard n]` | Runs the per-target SQL and featurisation, and writes `cache/<tid>.npz` plus `cache/meta.json` |
| `gate --tids ...` | Runs the original fit function, verbatim, on cached targets and compares the result with `../evidence/censoring_raw.json` |
| `sweep --lam L` | Runs the corrected fit at a fixed λ on all targets and writes `s4_lam_<L>.json` |
| `sweep --orig` | Runs the original fit on all targets, with fitted σ logged, and writes `s4_orig_full.json` |
| `sweep --tuned` | Runs the corrected fit with λ chosen by inner 5-fold CV and writes `s4_tuned.json` (details below) |
| `gradcheck` | Runs `scipy.optimize.check_grad` on the corrected objective and writes `gradcheck.json` |
| `summarise` | Writes `s4_sensitivity_summary.json` |

The corrected fit makes two changes, and leaves the objective, initialisation and tolerances unchanged:

1. The censored term uses `scipy.special.log_ndtr`, and the Mills ratio is computed as `exp(log φ − log Φ)`.
2. log σ has an L-BFGS-B bound of [−5, 5], and there is no clip inside the objective.

The tuned mode works as follows:

- λ is chosen separately for each target, seed and arm.
- The grid is {1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10}.
- The 5-fold CV runs on that arm's training rows only.
- The score is the pooled RMSE on the held-out *exact* training rows. Censored rows are never scored.
- After selection, the model is refit on all of the arm's training rows at the chosen λ and scored on the exact-only test set.
- The inner-fold permutation seeds are 10000+seed for the naive and Tobit arms (which share training rows), 20000+seed for exact-only and 30000+seed for size-matched.

The summary reports, for each run:

- the four contrasts (naive−exact, matched−exact, tobit−exact, tobit−naive), each with its mean, bootstrap CI and win counts
- the bootstrap method of `make_ledger.py`: 10,000 resamples over targets, `default_rng(20260904)`, percentiles 2.5 and 97.5
- the Spearman ρ against censored fraction
- strata by exact training-set size (<1024, 1024–1500, ≥1500)
- per-arm fit diagnostics
- for the tuned run, the distribution of the chosen λ

### Other scripts

| script | what it does |
|---|---|
| `extras.py` | Runs the regression check over all fits and compares the corrected and original fits at λ=1e-3. It also reports the targets with σ at its bound and the contrasts at the best λ per arm, and writes `s4_sensitivity_extras.json` |
| `run_all.sh` | Runs the fixed-λ sweep chain |

## Outputs

| file | contents |
|---|---|
| `gate_result.json` | Regression gate. The original fit reproduces the published per-seed RMSEs exactly |
| `s4_lam_*.json`, `s4_orig_full.json`, `s4_tuned.json` | Per target and per seed: RMSE for each of the 4 arms, training-set sizes, and per-fit info (convergence, iterations, gradient norm, log σ, σ-at-bound flag, censored rows with z < −7, and the chosen λ with its CV scores for the tuned run) |
| `s4_tuned_subset.*` | An initial three-target timing run of the tuned mode. Its records were reused in the full tuned run |
| `s4_sensitivity_summary.json`, `s4_sensitivity_extras.json` | Aggregate results |

## Independent check

`independent/` holds a separately written re-implementation of the same analysis, with its own README.
