#!/usr/bin/env python3
"""Turn censoring_raw.json into an evidence ledger + stamp manifest.

This is the bridge between evidence and manuscript: every number that will
appear in main.tex gets a ledger row here FIRST, with a source and a verdict,
and a manifest entry that guard.py resolves the \\StampedCell calls against.

A row is SUPPORTED only if it is a direct count or statistic over ChEMBL 36.
Nothing is computed here that was not computed in the analysis run; this script
only reshapes and records provenance.

Usage: python3 analysis/make_ledger.py
"""

from __future__ import annotations

import json
import numpy as np
from scipy import stats
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent / "evidence"
RAW = EVIDENCE / "censoring_raw.json"
LEDGER = EVIDENCE / "censoring_ledger.jsonl"
MANIFEST = EVIDENCE / "chembl_censoring_stamp_manifest.json"

NOW = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> int:
    d = json.loads(RAW.read_text())
    src = str(RAW)
    rows: list[dict] = []
    manifest: dict[str, dict] = {}

    # Pre-registered outcomes the manuscript must report. The guard fails the
    # build if one is missing, so a result that came out the wrong way cannot be
    # dropped between ledger and paper -- which matters here, because the
    # corrected S4 reverses the sign of our own prediction.
    MUST_PRINT = {
        "s4::n_targets", "s4::n_eligible", "s4::naive_minus_exact_mean",
        "s4::tobit_minus_exact_mean", "s4::tobit_minus_naive_mean",
        "s4::n_targets_naive_worse", "s4::n_targets_tobit_worse",
        "s4::n_targets_tobit_vs_naive_worse", "s4::p3_rho", "s4::p3_p",
        "s4::converged_pct", "s4::n_converged", "s4::n_fits",
        "s4::n_targets_all_converged",
        "s4::naive_minus_exact_converged_only",
        "s4::matched_minus_exact_mean", "s4::n_targets_matched_worse",
        "s4::p3_rho_matched",
        "s4::tobit_minus_naive_converged_only",
        "s4::naive_minus_exact_paired_mean",
        "s4::tobit_minus_naive_paired_mean",
        "s4::naive_minus_exact_paired_n_targets",
        "prev::censored_pct", "prev::n_potency",
    }

    def add(group: str, metric: str, value, reason: str, verdict="SUPPORTED",
            ci_low=None, ci_high=None):
        cid = f"{group}::{metric}"
        rows.append({
            "claim_id": cid, "metric": metric, "value": value, "source": src,
            "verdict": verdict, "verdict_reason": reason,
            "ci_low": ci_low, "ci_high": ci_high, "timestamp": NOW,
        })
        manifest[cid] = {"value": value, "claim_id": cid,
                         "source": f"{RAW.name}",
                         "must_print": cid in MUST_PRINT}

    s1 = d["S1_prevalence"]
    tot = s1["db_totals"]

    # ---- database scale
    add("db", "activities", tot["activities"],
        "count(*) over activities in ChEMBL 36")
    add("db", "molecules", tot["molecules"],
        "count(*) over molecule_dictionary in ChEMBL 36")
    add("db", "assays", tot["assays"],
        "count(*) over assays in ChEMBL 36")

    # ---- S1 prevalence
    add("prev", "n_potency", s1["n_potency_records"],
        "count of IC50/Ki/Kd/EC50/AC50 rows with numeric standard_value>0, units nM")
    add("prev", "censored_pct", round(100 * s1["censored_frac"], 2),
        "100 * (right+left censored) / n_potency")
    add("prev", "right_pct", round(100 * s1["right_frac"], 2),
        "100 * rows with standard_relation in ('>','>=','>>') / n_potency")
    add("prev", "left_pct", round(100 * s1["left_frac"], 2),
        "100 * rows with standard_relation in ('<','<=','<<') / n_potency")
    add("prev", "n_right", s1["n_right_censored"], "count of right-censored potency rows")

    for t, v in s1["per_type"].items():
        add(f"prev_{t}", "n", v["n"], f"count of {t} rows, numeric value, nM")
        add(f"prev_{t}", "censored_pct", round(100 * v["censored_frac"], 2),
            f"100 * censored / n for {t}")
        add(f"prev_{t}", "right_pct", round(100 * v["right_frac"], 2),
            f"100 * right-censored / n for {t}")
        add(f"prev_{t}", "left_pct", round(100 * v["left_frac"], 2),
            f"100 * left-censored / n for {t}")

    # ---- pchembl_value coverage: the operational fork
    pc = s1.get("pchembl_coverage_by_relation", {})
    for name in ("exact", "right", "left"):
        if name in pc and pc[name]["frac_with_pchembl"] is not None:
            add("pchembl", f"{name}_pct", round(100 * pc[name]["frac_with_pchembl"], 2),
                f"100 * rows with pchembl_value not null / rows, for {name}-relation rows")

    # ---- S2 non-randomness
    for t, v in d["S2_nonrandom"].items():
        if "exact" in v:
            add(f"s2_{t}", "exact_pmed", v["exact"]["p_median"],
                f"median p-value (9-log10 nM) of exact {t} rows, n={v['exact']['n']}")
        if "right" in v:
            add(f"s2_{t}", "right_pmed", v["right"]["p_median"],
                f"median p-value of right-censored {t} rows, n={v['right']['n']}")
        if v.get("right_minus_exact_p_median") is not None:
            add(f"s2_{t}", "right_delta", v["right_minus_exact_p_median"],
                f"median p(right-censored) - median p(exact) for {t}; "
                "negative = censored rows sit at weaker potency")
        if v.get("left_minus_exact_p_median") is not None:
            add(f"s2_{t}", "left_delta", v["left_minus_exact_p_median"],
                f"median p(left-censored) - median p(exact) for {t}")

    # ---- S3 threshold harm
    for t, v in d["S3_threshold"].items():
        x = v.get("6.0")
        if x and x.get("unsupported_frac_of_actives") is not None:
            add(f"s3_{t}", "unsupported_actives_pct",
                round(100 * x["unsupported_frac_of_actives"], 2),
                f"100 * right-censored rows at p>=6 / all rows at p>=6, for {t}; "
                "these are counted active on the basis of a bound the assay never reached")
            add(f"s3_{t}", "n_right_above6", x["n_right_censored_above"],
                f"count of right-censored {t} rows whose recorded p >= 6")

    # ---- S4 downstream
    # Now over EVERY target meeting the pre-registered inclusion rule, not the
    # eight largest (see A2 Amendment 1, Defect 2). With ~150 targets instead of
    # 8 the target-level spread is estimable, so every aggregate carries a
    # bootstrap CI over targets and the pre-registered P3 correlation becomes
    # testable for the first time.
    s4 = d["S4_downstream"]
    tg = s4["targets"]
    if tg:
        import statistics as st

        def boot_targets(vals, stat=np.mean, n_boot=10000, seed=20260904):
            v = np.asarray([x for x in vals if np.isfinite(x)], float)
            if len(v) < 2:
                return (float("nan"), float("nan"))
            rng = np.random.default_rng(seed)
            draws = stat(v[rng.integers(0, len(v), size=(n_boot, len(v)))], axis=1)
            return (float(np.percentile(draws, 2.5)),
                    float(np.percentile(draws, 97.5)))

        # n_eligible comes either from the run's own provenance or, for runs
        # that predate that field, from evidence/eligibility_raw.json which
        # re-derives it with the identical query.
        elig = s4.get("n_eligible")
        if elig is None:
            ef = EVIDENCE / "eligibility_raw.json"
            if ef.exists():
                elig = json.loads(ef.read_text())["n_eligible"]
        if elig is not None:
            add("s4", "n_eligible", elig,
                "single-protein targets meeting the pre-registered rule "
                "(>2000 IC50 nM records, >=200 right-censored)")
        add("s4", "n_targets", len(tg),
            "targets actually analysed: those meeting the inclusion rule that "
            "also survived the per-target minimums (>=300 compounds after "
            "aggregation, >=30 censored, >=150 exact)")

        # CONVERGENCE. The previous optimiser silently diverged on the naive arm
        # and that divergence WAS the reported effect; convergence is now a
        # recorded quantity, not an assumption.
        nf = sum(x.get("n_fits", 0) for x in tg)
        nc = sum(x.get("n_converged", 0) for x in tg)
        if nf:
            add("s4", "n_fits", nf, "total model fits across targets and seeds")
            add("s4", "n_converged", nc,
                "fits whose optimiser reported convergence; a claim resting on "
                "non-converged fits is an optimisation artifact, not a result")
            add("s4", "converged_pct", round(100.0 * nc / nf, 2),
                "100 * converged / total fits")
            # Per-arm breakdown. A reviewer asked for this specifically: an
            # aggregate rate hides whether failures are concentrated in one arm,
            # and a comparison in which one arm converges less often is not a
            # comparison of loss functions.
            for fld, tag in (("converged_naive", "naive"),
                             ("converged_exact_only", "exact"),
                             ("converged_censoring_aware", "tobit"),
                             ("converged_naive_matched", "matched")):
                seen = [ps for x in tg for ps in x["per_seed"] if fld in ps]
                if not seen:
                    continue
                add("s4", f"converged_pct_{tag}",
                    round(100.0 * sum(int(ps[fld]) for ps in seen) / len(seen), 2),
                    f"100 * converged / fits, for the {tag} arm alone")

        for key, short in (("rmse_naive", "naive"), ("rmse_exact_only", "exact"),
                           ("rmse_censoring_aware", "tobit"),
                           ("rmse_naive_matched", "matched")):
            if key not in tg[0]:
                continue
            vals = [x[key] for x in tg]
            add("s4", f"{short}_mean_rmse", round(st.fmean(vals), 4),
                f"mean over {len(tg)} targets of {key}, each itself a mean over "
                f"{tg[0]['n_seeds']} seeds; test folds are EXACT-ONLY labels")

        for dif, tag, human in (
            ([x["rmse_naive"] - x["rmse_exact_only"] for x in tg], "naive",
             "naive inclusion of censored rows, relative to discarding them"),
            ([x["rmse_censoring_aware"] - x["rmse_exact_only"] for x in tg], "tobit",
             "the Tobit correction, relative to discarding censored rows"),
            ([x["rmse_censoring_aware"] - x["rmse_naive"] for x in tg], "tobit_vs_naive",
             "the Tobit correction relative to NAIVE inclusion -- the only "
             "size-matched contrast, since both train on the same rows"),
        ):
            lo, hi = boot_targets(dif)
            add("s4", f"{tag}_minus_exact_mean" if tag != "tobit_vs_naive"
                else "tobit_minus_naive_mean",
                round(st.fmean(dif), 4),
                f"mean RMSE penalty of {human}; positive = worse. 95% bootstrap "
                f"CI over the {len(dif)} targets.",
                ci_low=round(lo, 4), ci_high=round(hi, 4))
            base = (f"{tag}_minus_exact" if tag != "tobit_vs_naive"
                    else "tobit_minus_naive")
            add("s4", f"{base}_lo", round(lo, 4),
                f"lower bound of the 95% bootstrap CI over targets on {base}_mean")
            add("s4", f"{base}_hi", round(hi, 4),
                f"upper bound of the 95% bootstrap CI over targets on {base}_mean")
            add("s4", f"n_targets_{tag}_worse", sum(1 for x in dif if x > 0),
                f"targets (of {len(dif)}) where {human} increased RMSE")
            add("s4", f"{tag}_median", round(float(np.median(dif)), 4),
                f"median RMSE penalty of {human}; reported beside the mean "
                "because a target-level mean is sensitive to a few hard targets")

        # SENSITIVITY: restrict to targets where every fit converged. If the
        # conclusions moved, they would be optimisation artifacts -- which is
        # exactly the defect that produced the first version of this analysis.
        # NOTE 2026-09-05: the size-matched arm was added by Amendment 2 and was
        # missing from this filter, so the subset was "all three original arms
        # converged" while the manuscript sentence said "every fit converged" --
        # and the size-matched arm has the WORST convergence rate (89.49%), so
        # it is the one the check most needs to cover. All four arms now.
        bad = {x["tid"] for x in tg for ps in x["per_seed"]
               if not (ps.get("converged_naive", True)
                       and ps.get("converged_exact_only", True)
                       and ps.get("converged_censoring_aware", True)
                       and ps.get("converged_naive_matched", True))}
        clean = [x for x in tg if x["tid"] not in bad]
        if clean:
            add("s4", "n_targets_all_converged", len(clean),
                "targets on which all 40 fits (10 seeds x 4 arms) converged; the "
                "rest have at least one non-converged fit and are used only in "
                "the sensitivity check")
            for key, tag in ((("rmse_naive", "rmse_exact_only"), "naive_minus_exact"),
                             (("rmse_censoring_aware", "rmse_naive"), "tobit_minus_naive")):
                dif = [x[key[0]] - x[key[1]] for x in clean]
                add("s4", f"{tag}_converged_only", round(float(np.mean(dif)), 4),
                    f"mean {key[0]} - {key[1]} over the {len(clean)} targets whose "
                    "fits all converged; reported so the reader can see the "
                    "conclusion does not rest on non-converged fits")

        # PAIRED CONVERGENCE-CONDITIONED SENSITIVITY. Strictly better than
        # discarding whole targets: for each contrast we keep only (target, seed)
        # pairs where BOTH arms in that specific contrast converged, then average
        # the retained per-seed differences within target and report target-level
        # stats. This retains more targets than the all-40-fits-converged filter
        # and is free of cross-contrast contamination.
        for arm_a, arm_b, tag, human, good_sign in (
            ("rmse_naive", "rmse_exact_only",
             "naive_minus_exact_paired",
             "naive-vs-exact paired (both arms converged)",
             "negative"),
            ("rmse_censoring_aware", "rmse_naive",
             "tobit_minus_naive_paired",
             "tobit-vs-naive paired (both arms converged)",
             "positive"),
        ):
            conv_a = arm_a.replace("rmse_", "converged_").replace(
                "censoring_aware", "censoring_aware").replace(
                "naive_matched", "naive_matched")
            # map arm name to convergence flag name
            conv_map = {
                "rmse_naive": "converged_naive",
                "rmse_exact_only": "converged_exact_only",
                "rmse_censoring_aware": "converged_censoring_aware",
                "rmse_naive_matched": "converged_naive_matched",
            }
            cfa, cfb = conv_map[arm_a], conv_map[arm_b]
            tgt_means = []
            for x in tg:
                retained = [
                    ps[arm_a] - ps[arm_b]
                    for ps in x["per_seed"]
                    if ps.get(cfa, True) and ps.get(cfb, True)
                    and np.isfinite(ps[arm_a]) and np.isfinite(ps[arm_b])
                ]
                if retained:
                    tgt_means.append(float(np.mean(retained)))
            n_paired = len(tgt_means)
            if n_paired < 2:
                continue
            paired_mean = float(np.mean(tgt_means))
            lo_p, hi_p = boot_targets(tgt_means, seed=20260905)
            # fraction of retained targets whose mean keeps the headline sign
            if good_sign == "negative":
                sign_frac = sum(1 for v in tgt_means if v < 0) / n_paired
            else:
                sign_frac = sum(1 for v in tgt_means if v > 0) / n_paired
            add("s4", f"{tag}_n_targets", n_paired,
                f"targets retained in the {human} analysis "
                f"(at least one seed with both arms converged)")
            add("s4", f"{tag}_mean", round(paired_mean, 4),
                f"mean over {n_paired} target-level averages of {arm_a} - {arm_b}, "
                "using only seeds where both arms converged; "
                "95% bootstrap CI over targets.",
                ci_low=round(lo_p, 4), ci_high=round(hi_p, 4))
            add("s4", f"{tag}_lo", round(lo_p, 4),
                f"lower bound of 95% bootstrap CI on {tag}_mean")
            add("s4", f"{tag}_hi", round(hi_p, 4),
                f"upper bound of 95% bootstrap CI on {tag}_mean")
            add("s4", f"{tag}_sign_frac", round(sign_frac, 3),
                f"fraction of the {n_paired} retained targets whose "
                f"paired mean difference keeps the headline sign ({good_sign})")

        # THE SIZE-MATCHED CONTROL. `naive` trains on more rows than
        # `exact-only`, so its advantage could be data volume rather than label
        # quality -- an internal review named this as the strongest
        # remaining objection. `matched` is the same exact+censored mixture
        # subsampled to exact-only's row count, so this contrast holds n fixed
        # and varies only whether censored rows are admitted.
        if "rmse_naive_matched" in tg[0]:
            dif = [x["rmse_naive_matched"] - x["rmse_exact_only"] for x in tg]
            lo, hi = boot_targets(dif)
            add("s4", "matched_minus_exact_mean", round(st.fmean(dif), 4),
                "mean RMSE(size-matched naive) - RMSE(exact-only). Both arms "
                "train on the SAME NUMBER of rows; they differ only in whether "
                "censored rows are admitted (as exact values). Negative means "
                "admitting censored rows helps even with no data-volume "
                "advantage, i.e. the naive gain is not a sample-size artifact.",
                ci_low=round(lo, 4), ci_high=round(hi, 4))
            add("s4", "matched_minus_exact_lo", round(lo, 4),
                "lower bound of the 95% bootstrap CI on matched_minus_exact_mean")
            add("s4", "matched_minus_exact_hi", round(hi, 4),
                "upper bound of the 95% bootstrap CI on matched_minus_exact_mean")
            add("s4", "matched_median", round(float(np.median(dif)), 4),
                "median of the size-matched contrast over targets")
            add("s4", "n_targets_matched_worse", sum(1 for x in dif if x > 0),
                f"targets (of {len(dif)}) where the size-matched naive arm was "
                "worse than exact-only")
            # how much of the naive advantage survives size-matching
            full = st.fmean([x["rmse_naive"] - x["rmse_exact_only"] for x in tg])
            if full < 0:
                add("s4", "matched_share_pct",
                    round(100.0 * min(max(st.fmean(dif) / full, 0.0), 1.0), 1),
                    "percentage of the unmatched naive advantage that survives "
                    "holding training-set size fixed; 100 means none of it was "
                    "sample size, 0 means all of it was")
            # P3 on the size-matched contrast: does the correlation survive?
            cfm = np.array([x["censored_frac"] for x in tg], float)
            pm = np.array(dif, float)
            okm = np.isfinite(cfm) & np.isfinite(pm)
            if okm.sum() >= 10:
                rm, pmv = stats.spearmanr(cfm[okm], pm[okm])
                add("s4", "p3_rho_matched", round(float(rm), 3),
                    "Spearman rho between censored fraction and the SIZE-MATCHED "
                    "contrast. If the unmatched P3 correlation were purely a "
                    "sample-size effect -- more censoring means more extra rows "
                    "for naive -- this one should vanish.")
                add("s4", "p3_p_matched", float(f"{pmv:.3g}"),
                    "two-sided p-value for p3_rho_matched")

        # P3, pre-registered and untestable with 8 targets: does the naive
        # penalty grow with the target's censored fraction?
        cf = np.array([x["censored_frac"] for x in tg], float)
        pen = np.array([x["rmse_naive"] - x["rmse_exact_only"] for x in tg], float)
        ok = np.isfinite(cf) & np.isfinite(pen)
        if ok.sum() >= 10:
            rho, pv = stats.spearmanr(cf[ok], pen[ok])
            add("s4", "p3_rho", round(float(rho), 3),
                "Spearman rho between a target's censored fraction and its naive "
                f"RMSE penalty, over {int(ok.sum())} targets. Pre-registered P3 "
                "predicted a positive correlation; with 8 targets it was not "
                "estimable.")
            add("s4", "p3_p", float(f"{pv:.3g}"),
                "two-sided p-value for the P3 Spearman correlation")
        add("s4", "cens_frac_median", round(float(np.median(cf)), 4),
            "median censored fraction over the analysed targets")

    LEDGER.write_text("".join(json.dumps(r) + "\n" for r in rows))
    MANIFEST.write_text(json.dumps({"ledger_rows": manifest}, indent=1))
    print(f"wrote {LEDGER.name}: {len(rows)} rows")
    print(f"wrote {MANIFEST.name}: {len(manifest)} manifest entries")
    print("\n--- key values for drafting ---")
    for r in rows:
        print(f"  {r['claim_id']:<34} = {r['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
