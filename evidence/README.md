# Evidence ledger — conventions

Every numeric literal in `main.tex` resolves to a row in this folder's ledger,
except the `\SafeNum{...}` architecture constants and nominal parameters.
`guard.py` fails if it cannot resolve one.

## Ledger row schema

One JSONL file per ledger, one row per claim. Example from
`censoring_ledger.jsonl`:

```json
{
  "claim_id": "db::activities",
  "metric": "activities",
  "value": 24267312,
  "source": "/abs/path/to/evidence/censoring_raw.json",
  "verdict": "SUPPORTED",
  "verdict_reason": "count(*) over activities in ChEMBL 36",
  "ci_low": null,
  "ci_high": null,
  "timestamp": "2026-09-04T22:23:29Z"
}
```

Verdicts:
- `SUPPORTED` — the value traces to a real measurement, the real database or
  a cited source.
- `UNSUPPORTED` — blocks. The literal in `main.tex` must revert to
  `\ResultCell` or be removed.
- `NOT_APPLICABLE` — informational only.
- `CONTESTED` — needs review, and blocks release until resolved.

When a `claim_id` appears more than once, the later row supersedes the
earlier one.

## The stamp manifest

`chembl_censoring_stamp_manifest.json` maps each `\StampedCell{k1}{k2}{v}` key
to the value it must equal. A manifest entry flagged `must_print: true` is a
pre-registered outcome or headline claim, and the guard fails if the
manuscript omits it. `must_print: false` rows are recorded for audit only.

```json
{
  "ledger_rows": {
    "prev::n_potency": {
      "value": 5013576,
      "claim_id": "prev::n_potency",
      "source": "censoring_raw.json",
      "must_print": true
    }
  }
}
```

`guard.py` walks `main.tex` for every `\StampedCell{k1}{k2}{v}` and asserts
that `manifest.ledger_rows["k1::k2"].value == v`, with numeric tolerance. It
also asserts that the backing ledger row's verdict is `SUPPORTED`. Claim ids
that `make_figure.py` lists in `figure_claims.json` count as reported.
