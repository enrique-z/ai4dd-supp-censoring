# Supplementary Note — ChEMBL censoring

## Evidence files

The files in `evidence/` (censoring_ledger.jsonl, censoring_raw.json,
eligibility_raw.json, figure_claims.json,
chembl_censoring_stamp_manifest.json) are byte-identical to the outputs of the
original run. Their `source` and `db_path` fields contain absolute paths from
that run, of the form `/Users/apple/code/…`. These are kept as recorded so the
records match the run that produced them.

## Reproducing the analysis

Requirements: Python 3.10+, NumPy, SciPy, matplotlib, and a local SQLite copy
of ChEMBL 36 (`chembl_36.db`). By default the scripts look for the database at
`./data/chembl_36/chembl_36_sqlite/chembl_36.db`, relative to the directory
you run them from. `run_censoring.py` also accepts `--db PATH`.

Run the scripts from the repository root:

```bash
python3 analysis/run_censoring.py --db ./data/chembl_36/chembl_36_sqlite/chembl_36.db
                                    # S1-S4 -> evidence/censoring_raw.json (S4 is the long step)
python3 analysis/record_eligibility.py   # -> evidence/eligibility_raw.json (reads the default db path)
python3 analysis/make_ledger.py          # -> evidence/censoring_ledger.jsonl + stamp manifest
python3 analysis/make_figure.py          # Figure 1 -> fig_s4.pdf, evidence/figure_claims.json
python3 analysis/guard.py --selftest
```

`python3 analysis/run_censoring.py --quick` runs a short smoke test (S4 on
2 targets × 3 seeds instead of every eligible target × 10 seeds). It does not
reproduce the paper's S4 numbers. `make_figure.py` recomputes the three contrast
means and both Spearman correlations from the per-target rows and aborts if
they disagree with the ledger. The scripts overwrite files in `evidence/`, so
run them in a copy of the repository if you want to compare against the
shipped evidence.

To check a manuscript source against the evidence, place `main.tex` in the
directory that contains `evidence/` and run
`python3 analysis/guard.py --tex main.tex --strict`.

## Pre-registration

`A2_PREDECLARATION_2026-09-03.md` is the pre-registration cited in the paper.
For this release, internal workflow notes were removed from it. The claim,
predictions P1–P3, refutation rules R1–R4, and the substance of Amendments
1–4 (defects found, fixes, outcomes, and declined recommendations) are
unchanged. The first release of this repository carried a copy made before
Amendment 4. The current copy includes Amendment 4 and the two OUTCOME lines
it corrected. `A2_AMENDMENT_5_2026-10-24.md` records the post-hoc penalty
analysis added for the camera-ready version.
