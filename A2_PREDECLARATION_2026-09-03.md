# A2 — Pre-declaration · censored potency measurements in ChEMBL 36

**Venue:** NeurIPS 2026 Workshop AI4DD (non-archival, 5-page full-paper
track, deadline 2026-09-06 11:59 UTC)
**Written:** 2026-09-03, before the confirmatory S4 benchmark was analysed.

## Honesty note on the order of operations

The descriptive strata (S1–S3) are **census statistics over a fixed public
database**, not sampled estimates: they are exhaustive counts over ChEMBL 36 and
have no sampling error to be surprised by. Exploratory queries established their
magnitude before this file was written, and those values are stated below rather
than hidden — pre-registering a number I have already counted would be theatre.

What is genuinely pre-registered here is **S4**, the downstream modelling
experiment, whose outcome is unknown at the time of writing. A first attempt at
S4 selected zero targets because its query required `pchembl_value is not null`,
which excludes 100% of censored rows; that bug and its fix are recorded below.

## Background and the mechanism under study

ChEMBL stores each activity with a `standard_relation`. `'='` is an exact
determination. `'>'` means the assay never reached a response, so the true
concentration is **above** the recorded value and the compound is **weaker**
than recorded. `'<'` means the response saturated below the lowest concentration
tested, so the compound is **stronger** than recorded. A `'>'` or `'<'` row is a
censored observation: the true value lies in a half-open interval, not at the
recorded number.

Benchmark builds that read `standard_value` and discard `standard_relation`
convert "IC50 > 40,000 nM" into "IC50 = 40,000 nM".

## Census findings already counted (S1–S3, exhaustive, not pre-registered)

Over 5,013,576 potency records (IC50/Ki/Kd/EC50/AC50, numeric value, units nM):

| statistic | value |
|---|---|
| censored overall | **24.34%** (20.34% right, 3.99% left) |
| Kd censored | **54.57%** (51.25% right) |
| AC50 censored | 50.42% |
| EC50 / IC50 / Ki censored | 25.04% / 21.43% / 20.13% |
| median p-value, exact vs right-censored (IC50) | 6.22 vs 4.49 (**Δ −1.73 log units**) |
| same, Kd | 6.60 vs 4.52 (**Δ −2.08**) |
| `pchembl_value` populated, exact rows | 96.10% |
| `pchembl_value` populated, censored rows | **0.00%** |

The last two lines are the operational fork: a pipeline keyed on
`pchembl_value` silently drops every censored row — safe, but discards a quarter
of the data with no warning — whereas a pipeline keyed on `standard_value`
inherits the full bias.

## Pre-registered claim (S4)

Training a potency regressor on ChEMBL data with censored rows treated as exact
observations produces a model that is **worse on trustworthy held-out labels**
than either discarding those rows or modelling them as bounds.

## Numeric predictions and directions (S4)

| # | prediction | direction |
|---|---|---|
| P1 | RMSE(naive) > RMSE(exact-only) on exact-only held-out test data | naive worse |
| P2 | RMSE(censoring-aware Tobit) ≤ RMSE(exact-only) | Tobit at least as good |
| P3 | The naive penalty grows with the target's censored fraction | positive correlation |

## Refutation criteria

- **R1.** If mean RMSE(naive) ≤ mean RMSE(exact-only) across the selected
  targets, P1 is refuted and the paper reports that naive inclusion of censored
  rows is harmless — a genuinely useful scoping result, since it would license
  the common practice.
- **R2.** If Tobit does not beat naive on any target, the censoring-aware remedy
  is reported as ineffective rather than quietly dropped.
- **R3 (evaluation integrity).** The test set is **exact-only** in every
  condition. Censored labels cannot serve as test ground truth — their true
  values are unknown, so scoring against them would be scoring predictions
  against numbers nobody measured. Any result computed on a test set containing
  censored labels is inadmissible.
- **R4 (practical significance).** An RMSE difference below 0.05 log units is
  reported as null regardless of its sign or p-value.

## AMENDMENT 1 — 2026-09-03, after the first internal review

An internal review returned **reject, 3/10**, and it was right about three
things that are recorded here rather than quietly patched.

**Defect 1 — a truncated sample reported as a census statistic.** `s2_nonrandom`
carried `limit=500_000` with **no `ORDER BY`**, so the S2 medians summarised
whichever rows SQLite scanned first, while the manuscript presented them as
exhaustive counts over the release. The reviewer recomputed over all rows and
got a median IC50 shift of **-1.6982** against our reported **-1.7269**. The
qualitative effect survives; the description did not. The limit is removed and
S2 is now exhaustive. This is the most serious defect found: a number that was
not what the text said it was.

**Defect 2 — an undisclosed and unlicensed target selection.** The pre-registered
inclusion rule is ">2,000 IC50 records and >=200 right-censored, single
protein". The implementation then applied `order by c desc limit 40` followed by
`[:8]`, so the benchmark ran on the eight **largest** eligible targets. The rule
admits **159**. Nothing in the pre-registration licensed picking the biggest
eight, and the manuscript did not disclose it. S4 now runs on all 159.

**Defect 3 — arbitrary duplicate resolution.** Where several rows shared a
canonical SMILES the code kept whichever the scan returned first. Measurements
are now aggregated per compound: a compound with any exact measurement takes the
median of its exact values and counts as uncensored (a determination dominates a
bound); otherwise it takes the median of its bounds and counts as censored.

**Also fixed, no change to the mathematics:** `tobit_fit` computed the normal
CDF with `np.vectorize(math.erf)`, a Python-level loop inside all 300 gradient
iterations. Replaced with `scipy.special.erf`, verified bit-identical on a test
grid. This is what made a 159-target run feasible at all.

**R4 is binding and was being argued around.** R4 states that "an RMSE
difference below 0.05 log units is reported as null regardless of its sign or
p-value". The draft acknowledged that the naive penalty (0.0276) fell below the
threshold and then argued past it via a sign test, concluding "the effect is
real and small, not absent". That is the violation R4 exists to prevent. The
finding is reported as **null**, with the sign consistency noted as a secondary
observation that does not overturn the pre-registered rule.

**Minor inconsistency, also flagged and corrected.** The census table above
recorded `pchembl_value` coverage on exact rows as 96.10%. The database and the
manuscript both say **96.53%**. The pre-declaration figure came from an earlier
exploratory query and was never reconciled; the manuscript value is the correct
one and is the one backed by a ledger row.

**Novelty position corrected.** The reviewer named prior work the original
prior-art sweep missed. The Related Work section and the contribution claim are
revised accordingly; see the manuscript.

All S4 numbers in the manuscript are regenerated from the 159-target run. The
census (S1-S3) is unchanged apart from S2 becoming exhaustive.

### Outcome of the corrected run — recorded against the pre-registration

**Two of the three pre-registered predictions are refuted, and P1 is refuted in
the direction opposite to the one predicted. P2 is a null under R4 — see
Amendment 4, which corrects an earlier "refuted" verdict in this table.**

| | prediction | result |
|---|---|---|
| P1 | RMSE(naive) > RMSE(exact-only) | **reversed**: naive is BETTER by 0.0777 log units, CI [-0.1038, -0.0545], worse on only 45/158 targets |
| P2 | RMSE(Tobit) ≤ RMSE(exact-only) | **null, NOT refuted** (corrected in Amendment 4): Tobit is worse by 0.0301 vs exact-only, which is below the R4 floor of 0.05 and is therefore reported as null. The 0.1077 penalty vs naive is real and above the floor, but naive is not P2's comparator. |
| P3 | naive penalty grows with censored fraction | **inverted**: Spearman ρ = −0.567, p = 8e-15 — it shrinks |

**R1 fires as written.** R1 says: "If mean RMSE(naive) ≤ mean RMSE(exact-only)
across the selected targets, P1 is refuted and the paper reports that naive
inclusion of censored rows is harmless — a genuinely useful scoping result,
since it would license the common practice." The measured result is stronger
than "harmless": naive inclusion is *better*. The paper reports that, and does
not upgrade the claim beyond what R1 anticipated.

**R4 is now moot but was still violated in the draft.** The naive difference
(0.0777) exceeds the 0.05 practical-significance floor, so R4 no longer forces a
null verdict. The draft's earlier violation — arguing past R4 via a sign test
when the effect was 0.0276 — is recorded above and was fixed before this run
produced its numbers, not after.

### AMENDMENT 2 — the size-matched control, added after the second internal review

The second round of internal review (6/10, borderline) named one objection as
strongest: the naive-vs-exact-only contrast does not separate label noise from
sample size, because naive trains on ~20% more rows. It asked for
"fixed-total-size resampling ... compare naive and exact-only at matched
training-set sizes, repeatedly and per target".

We added exactly that arm: the same exact-plus-censored mixture, subsampled to
exact-only's row count. **The reviewer was right and our stated mechanism was
wrong.** At equal n the censored rows *hurt*, by +0.0465 log units
[0.0349, 0.0586] on 149/158 targets; 0% of the unmatched advantage survives
size-matching. The pre-registered P3 correlation flips from −0.567 (unmatched)
to +0.429 (size-matched, p = 1.9e-08) — the size-matched sign is the direction
P3 originally predicted.

The corrected reading is two effects of opposite sign: treating a bound as a
value damages a model and damages it more the more censored the target, while
deleting the rows damages it more still because a fifth of the training set goes
with them. The manuscript now reports the decomposition rather than the earlier
single mechanism ("censored rows carry information"), which is withdrawn.

**The reversal was caused by the optimiser, not by the target expansion.** This
is the single most important fact for anyone auditing this paper: refitting the
identical objective and gradient with L-BFGS instead of 300 fixed-step gradient
iterations flipped the sign of the headline. Convergence is now recorded
(5,906/6,320 fits, 93.45%, spread across arms rather than concentrated — 97.28%
naive, 96.71% Tobit, 90.32% exact-only, 89.49% size-matched), and the
sensitivity restricted to the 129 fully-converged targets gives −0.0617 and
+0.0973 — the same conclusions.

*(Corrected 2026-09-05: this sensitivity previously read 133 targets, −0.0637 and
+0.0986. That subset was filtered on the three original arms only — the
size-matched arm added by this amendment was missing from the filter, so
"every fit converged" excluded the arm with the WORST convergence rate (89.49%).
The filter now covers all four arms. Same conclusions, as before.)*

*(Corrected 2026-09-04: this paragraph previously read 4,492/4,740 fits at
94.77%. That was the three-arm count — 158 targets × 10 seeds × 3 arms — written
before this very amendment added the size-matched arm as a fourth, which takes
the denominator to 6,320. The manuscript always carried the four-arm figure;
only this file was stale.)*

### AMENDMENT 3 — 2026-09-04, presentation only, after the third internal review

The third round of internal review scored the size-matched version **7/10, weak
accept** (up from 6/10) and confirmed the round-2 objection is answered. It ran
no independent computations, so there was nothing to adjudicate numerically;
its recommendations were judgement calls. Three were acted on, and **none of
them touches the analysis** — no model was refit and no evidence file changed.

1. **A figure, ranked its highest score-gain per unit of compute.** The paper
   had none. Figure 1 plots the per-target RMSE differences for all three
   contrasts and the two pre-registered P3 correlations, from data already
   collected and previously reported only as aggregates. `make_figure.py`
   recomputes the three contrast means and both Spearman correlations from
   `S4_downstream.targets` and aborts if they disagree with the ledger; they
   agree. This is an independent reproduction of the headline numbers from the
   per-target rows, and it is the first one performed outside the original
   analysis script.
2. **Space for it**, per the same review's answer on what to cut: the
   threshold-classification subsection is compressed to a single paragraph, the
   two census tables are merged into one, and a duplicated explanation of why
   censoring marks the tails is stated once instead of twice.
3. **Scope discipline in the abstract and conclusion.** The review's framing
   answer was to keep the census as the lead but stop stating the modelling
   result unconditionally. Both now say "in this linear IC50 benchmark", which
   matches what Limitations already conceded. The conclusion's novelty claim is
   narrowed from "have not been quantified before" to "have not been quantified
   *release-wide* before", which is what Related Work actually argues given
   Fralish et al.'s 20% figure over 230 selected IC50 datasets.

Two recommendations were **declined** and are recorded as declined: an
independently implemented censored loss (the review's single top pick) and
scaffold-disjoint splits. Both are hours of compute against a deadline, and both
are already disclosed as limitations in the manuscript. The Tobit result
therefore remains a single-implementation finding and is claimed as such.

One further defect was found by the authors' own re-read rather than by the
review: section 4.4 said the unsupported actives were "3.47% of everything
called active", but the computed denominator is exact-plus-right-censored
records and excludes left-censored actives (148,372 of them for IC50). All five
endpoints were reproduced directly against ChEMBL 36 and match the ledger
exactly; the inclusive denominator would give 3.15% for IC50. The paper's figure
was therefore the conservative one — the error ran against us — but the prose
did not describe what was computed, and has been narrowed to match.

## Rival mechanisms

1. **The naive penalty is really a sample-size advantage in disguise.** Naive
   trains on more rows than exact-only, so if naive still loses, the effect is
   conservative. If naive *wins*, it may be winning on volume rather than label
   quality — separated by comparing against exact-only subsampled to naive's n.
2. **Target-specific idiosyncrasy.** Controlled by running ≥6 single-protein
   targets and reporting per-target results, never a pooled headline alone.
3. **Tobit's gain is an artifact of the linear model class.** Disclosed as a
   scope limitation: the claim is about the loss function, evaluated within one
   model class held fixed across all three conditions.

## Recorded bug and its fix (pre-analysis)

The first S4 target query required `act.pchembl_value is not null`. Because
ChEMBL populates `pchembl_value` for **0.00%** of censored rows, that predicate
excluded exactly the records the study is about, and the run selected zero
targets. Fixed by selecting on `standard_value` and restricting to
`target_type = 'SINGLE PROTEIN'`. The bug is retained as a comment in
`analysis/run_censoring.py` because it is itself an instance of the paper's
thesis: the convenient field silently removes the censored data.

## Exact ledger rows the run will produce

`evidence/censoring_raw.json` → `evidence/censoring_ledger.jsonl` +
`evidence/chembl_censoring_stamp_manifest.json`, with sections
`S1_prevalence` (per-type counts, pchembl coverage, db totals),
`S2_nonrandom` (p-scale medians/percentiles per relation per type),
`S3_threshold` (counts above p ∈ {5,6,7} by relation),
`S4_downstream` (per target: n_compounds, censored_frac, and mean ± sd RMSE over
10 seeds for naive / exact-only / censoring-aware).

## Assets consumed

- ChEMBL 36 SQLite (`chembl_36.db`), extracted 2026-09-03 from the distributed
  tarball (5,611,751,319 B → 29,739,835,392 B).
  Verified live: 24,267,312 activities, 2,878,135 compounds, 1,890,749 assays.

## Feasibility — the four checks

**CHECK 1 — positive-result reachability.** S1–S3 are exhaustive counts over a
5.0M-row census; there is no power question. S4 uses single-protein targets with
>2,000 IC50 records and >200 right-censored rows, giving test folds of several
hundred exact labels per seed over 10 seeds — sufficient to resolve the
pre-registered 0.05 log-unit threshold.

**CHECK 2 — fallback narrative.**
*Positive:* naive inclusion measurably degrades models → an actionable data-quality
result for CFP topic T2 ("noisy or censored assays").
*Negative:* naive inclusion is harmless → the census (S1–S3) still stands on its
own as a characterisation of a 24%-prevalent, strongly non-random data property
that the field's tooling handles inconsistently, and the negative S4 licenses
current practice. Both branches publishable.

**CHECK 3 — feature-disjointness.** `n/a` — single hypothesis, no composed
sub-hypotheses.

**CHECK 4 — outside-labeller status.** `outside`. All measurements are
third-party experimental determinations curated by EMBL-EBI. Neither the labels
nor the censoring flags originate from the authors. No disclosure caveat
required.

## CFP topic match

Primary: **T2 Learning under Scarce, Biased, and Shifting Data** — "noisy or
**censored assays**" is a named topic, matched literally.
Secondary: **T1** — "realistic benchmark design, … evidence standards,
reproducibility".

## Prior-art position

Censored-data methods (Tobit regression, survival analysis) are textbook
statistics, and the existence of `'>'` records in ChEMBL is known to
practitioners. What is not established in the ML-for-drug-discovery literature is
a quantitative census of the problem at ChEMBL scale with the downstream cost
measured. The paper claims the measurement and the cost, **not** the invention of
censored regression, and will say so explicitly to avoid overclaiming.

---

## AMENDMENT 4 — 2026-09-05, after the fourth internal review

**R4 was violated a second time, in the direction the first violation was
fixed.** Amendment 1 recorded that the draft had argued past R4 with a sign test
when the naive penalty was 0.0276. The confirmatory run then produced two more
sub-floor contrasts, and the manuscript did the same thing again:

| contrast | value | floor | what the draft said |
|---|---|---|---|
| size-matched minus exact-only | 0.0465 | 0.05 | "the censored labels **hurt**", "a small label-quality **cost**", sign test 149/158 |
| Tobit minus exact-only | 0.0301 | 0.05 | "it **costs** 0.0301", sign test 125/158, "**P2 is refuted**" |

R4 states: "An RMSE difference below 0.05 log units is reported as null
regardless of its sign or p-value." Both are below the floor. Under our own
rule both are nulls, and the reviewer was right to flag it.

**Correction applied to the manuscript and to the OUTCOME table above.**

1. **P2 is NOT refuted.** P2 predicted RMSE(Tobit) ≤ RMSE(exact-only). The
   measured gap is 0.0301 — a practical null under R4, and a null is consistent
   with the prediction. The OUTCOME table's verdict of "refuted" reached for the
   Tobit-vs-**naive** contrast (0.1077), but naive is not P2's comparator. That
   was a goalpost move and is withdrawn. Two of the three pre-registered
   predictions are refuted (P1 reversed, P3 inverted); P2 is reported as a null.
2. **The size-matched arm is a null.** What survives is the decomposition, not a
   label-quality cost: 0.0% of the unmatched advantage survives size-matching,
   so the naive win is attributable to volume. At equal n the labels are a wash.
   The manuscript no longer says the censored labels "hurt".
3. **What still clears the floor** is Tobit vs naive (0.1077, worse on 156/158)
   — the censoring-aware remedy loses to the shortcut it exists to beat. R2
   applies and that claim is unchanged.
4. **Limitations** now names the un-tuned Tobit hyperparameters explicitly
   (λ = 1e-3 and a 300-step gradient budget, both fixed a priori) and the
   homoscedastic Gaussian error model, since "Tobit is worse than naive" is the
   one Tobit claim still above the floor and it is the claim most exposed to
   "your Tobit was simply under-fit".

No number changed; every stamped cell is the value it was. This is a verdict
correction, not a re-analysis.

**Rejected from the same review:** the reviewer's proposed reframing that the
bounds therefore have "immense virtue" and are "practically as good as an exact
measurement". A sub-floor difference means we cannot resolve a difference at the
0.05 scale; it is not evidence of equivalence, and asserting it would be the
same overclaim in the opposite direction. The manuscript says the labels are a
wash at equal volume and stops there.
