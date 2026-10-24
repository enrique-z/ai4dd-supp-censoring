# Analysis scripts

The scripts locate `evidence/` as `../evidence`, relative to this directory.
Run them from the repository root. The exact commands and their order are in
`SUPPLEMENTARY_NOTE.md`.

| Script | Reads | Writes |
|--------|-------|--------|
| `run_censoring.py` | ChEMBL 36 (read-only) | `evidence/censoring_raw.json` |
| `record_eligibility.py` | ChEMBL 36 (read-only) | `evidence/eligibility_raw.json` |
| `make_ledger.py` | the two raw files above | `evidence/censoring_ledger.jsonl`, `evidence/chembl_censoring_stamp_manifest.json` |
| `make_figure.py` | raw evidence + ledger | `fig_s4.pdf`, `evidence/figure_claims.json` |
| `guard.py` | manuscript + ledger + manifest | exit code and report |

## Conventions every script follows

1. Read real data only: ChEMBL 36, opened read-only. No rows are synthesised.
2. Every number that reaches the manuscript first gets a ledger row with a
   `source` and a `verdict`, using the schema in `evidence/README.md`.
3. The stamp manifest maps every `\StampedCell{k1}{k2}{v}` key to its value.
   `guard.py` checks the manuscript against it.
4. Be deterministic: given the same inputs, produce the same ledger values.
5. A row whose verdict is not `SUPPORTED` is kept for inspection, never
   silently dropped.

Run `python3 analysis/guard.py --selftest` to check the guard itself on temporary
fixtures. See `evidence/README.md` for the guard's error classes.
