# A quarter of ChEMBL's potency labels are bounds — and deleting them can cost more than keeping them

Supplementary code, pre-registration and evidence ledger.

**Authors:** Enrique Zueco, David Scott Lewis, Haley Yi
**Affiliation:** AIXC Bio, AIXC Research, Zaragoza, Spain
**Contact:** reports@aixcresearch.com
**Venue:** AI for Drug Discovery (AI4DD) workshop, NeurIPS 2026

## What this is

Analysis code, the pre-registration document, and an evidence ledger that backs every quantitative claim in the paper.

## Directory layout

| Path | Contents |
|------|----------|
| `A2_PREDECLARATION_2026-09-03.md` | Pre-registration (predictions P1–P3, refutation rules R1–R4) and its recorded amendments |
| `A2_AMENDMENT_5_2026-10-24.md` | Camera-ready amendment: the post-hoc penalty analysis and how it is reported against the pre-registered rules |
| `analysis/run_censoring.py` | Main analysis: S1 prevalence, S2 non-randomness, S3 threshold harm, S4 downstream benchmark, all on ChEMBL 36 → `evidence/censoring_raw.json` |
| `analysis/record_eligibility.py` | Re-derives the number of targets meeting the S4 inclusion rule → `evidence/eligibility_raw.json` |
| `analysis/make_ledger.py` | Builds the evidence ledger and stamp manifest from the raw evidence |
| `analysis/make_figure.py` | Figure 1 (per-target S4 decomposition): recomputes from the raw evidence, asserts against the ledger, then draws |
| `analysis/guard.py` | Integrity guard: checks every stamped number in the manuscript against the stamp manifest and ledger |
| `penalty_sensitivity/` | Post-hoc S4 analysis (paper §4.4 and Appendix B): corrected Tobit fit and ridge-penalty sweep including a cross-validated penalty, with outputs and an independent re-implementation (`independent/`); see its `README.md` |
| `evidence/` | Raw JSON outputs, evidence ledger (`censoring_ledger.jsonl`) and stamp manifest |
| `evidence/README.md` | Ledger and manifest conventions |
| `analysis/README.md` | How the scripts fit together and how to rerun them |
| `SUPPLEMENTARY_NOTE.md` | Provenance notes and reproduction instructions |
| `MANIFEST.txt` | Size and SHA-256 of every file except `README.md` and `MANIFEST.txt` |
| `refs.bib` | Bibliography |

Every pre-registered number in the manuscript resolves from the evidence ledger through the stamp manifest; the post-hoc penalty-sensitivity numbers come from `penalty_sensitivity/`. See `SUPPLEMENTARY_NOTE.md` for provenance and reproduction.
