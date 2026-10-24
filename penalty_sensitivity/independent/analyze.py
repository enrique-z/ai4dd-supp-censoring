"""Summarise ridge_cv results; validate against the paper's censoring_raw.json."""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
res = json.load(open(sys.argv[1]))
RAW = json.load(open(HERE.parent.parent / "evidence" / "censoring_raw.json"))
orig = {t["tid"]: t for t in RAW["S4_downstream"]["targets"]}
GRID = np.array(res["grid"]); R = res["results"]; S = res["n_seeds"]
out = {"source": sys.argv[1], "n_targets": len(R), "n_seeds": S}

# ---- 1. validation against the original run
size_mismatch, dn, de, dm = 0, [], [], []
for r in R:
    o = orig[r["tid"]]["per_seed"]
    for p in r["per_seed"]:
        q = o[p["seed"]]
        size_mismatch += (p["n_train_exact_only"] != q["n_train_exact_only"]) + (p["n_train_all"] != q["n_train_all"])
        dn.append(p["naive"]["rmse_implied"] - q["rmse_naive"])
        de.append(p["exact_only"]["rmse_implied"] - q["rmse_exact_only"])
        dm.append(p["matched"]["rmse_implied"] - q["rmse_naive_matched"])
abs_ = lambda v: {"median_abs": float(np.median(np.abs(v))), "p95_abs": float(np.percentile(np.abs(v), 95)),
                  "max_abs": float(np.max(np.abs(v)))}
out["validation"] = {"targets_missing_from_orig": [r["tid"] for r in R if r["tid"] not in orig],
                     "orig_targets_not_run": len(set(orig) - {r["tid"] for r in R}),
                     "size_mismatches": int(size_mismatch),
                     "implied_vs_orig_rmse_naive": abs_(dn), "implied_vs_orig_rmse_exact": abs_(de),
                     "implied_vs_orig_rmse_matched": abs_(dm)}

rng = np.random.default_rng(12345)
def summ(v):
    v = np.asarray(v)
    bs = v[rng.integers(0, len(v), (10_000, len(v)))].mean(1)
    return {"mean": round(float(v.mean()), 5), "ci95": [round(float(np.percentile(bs, 2.5)), 5),
            round(float(np.percentile(bs, 97.5)), 5)], "n_worse": int((v > 0).sum()),
            "n_better": int((v < 0).sum()), "n": int(len(v))}

def per_target(key_fn):
    return np.array([np.mean([key_fn(p) for p in r["per_seed"]]) for r in R])

def block(getter, name, idxs=None):
    n_minus_e = per_target(lambda p: getter(p, "naive") - getter(p, "exact_only"))
    m_minus_e = per_target(lambda p: getter(p, "matched") - getter(p, "exact_only"))
    sel = np.arange(len(R)) if idxs is None else idxs
    return {"naive_minus_exact": summ(n_minus_e[sel]), "matched_minus_exact": summ(m_minus_e[sel])}

g_cv = lambda p, a: p[a]["rmse_cv"]
g_imp = lambda p, a: p[a]["rmse_implied"]
g_cvx = lambda p, a: p[a]["rmse_cvx"] if a != "exact_only" else p[a]["rmse_cv"]
out["cv_tuned"] = block(g_cv, "cv")
out["cv_tuned_naive_validated_on_exact_rows"] = block(g_cvx, "cvx")
out["implied_alpha_original_lam1e-3"] = block(g_imp, "implied")
# original JSON numbers for reference
out["original_json"] = {
    "naive_minus_exact": summ([orig[r["tid"]]["rmse_naive"] - orig[r["tid"]]["rmse_exact_only"] for r in R]),
    "matched_minus_exact": summ([orig[r["tid"]]["rmse_naive_matched"] - orig[r["tid"]]["rmse_exact_only"] for r in R])}

# implied-alpha distribution and a single global fixed alpha
imp = {a: np.array([p[a]["alpha_implied"] for r in R for p in r["per_seed"]]) for a in ("naive", "exact_only", "matched")}
out["implied_alpha_quantiles"] = {a: [float(np.percentile(v, q)) for q in (5, 25, 50, 75, 95)] for a, v in imp.items()}
out["implied_multi_root_fits"] = {a: int(sum(p[a]["n_roots"] > 1 for r in R for p in r["per_seed"])) for a in imp}
med = float(np.median(np.concatenate(list(imp.values()))))
j = int(np.argmin(np.abs(np.log10(GRID) - np.log10(med))))
out["fixed_global_alpha"] = {"alpha": float(GRID[j]), "median_implied": med,
                             **block(lambda p, a: p[a]["rmse_grid"][j], "fixed")}
out["fixed_alpha_sweep"] = {f"{GRID[k]:.4g}": {kk: vv["mean"] for kk, vv in
                            block(lambda p, a, k=k: p[a]["rmse_grid"][k], "g").items()} |
                            {"naive_worse": block(lambda p, a, k=k: p[a]["rmse_grid"][k], "g")["naive_minus_exact"]["n_worse"]}
                            for k in range(len(GRID))}
cva = {a: np.array([p[a]["alpha_cv"] for r in R for p in r["per_seed"]]) for a in ("naive", "exact_only", "matched")}
out["cv_alpha_quantiles"] = {a: [float(np.percentile(v, q)) for q in (5, 25, 50, 75, 95)] for a, v in cva.items()}
out["cv_edge_hits"] = {a: int(sum(p[a]["cv_edge"] for r in R for p in r["per_seed"])) for a in cva}
out["cv_vs_implied_exact_only_rmse_gain"] = summ(per_target(lambda p: p["exact_only"]["rmse_cv"] - p["exact_only"]["rmse_implied"]))

# ---- stratified by n_exact_train
ne = np.array([r["per_seed"][0]["n_train_exact_only"] for r in R])
strata = {"<1024": np.where(ne < 1024)[0], "1024-1500": np.where((ne >= 1024) & (ne < 1500))[0],
          ">=1500": np.where(ne >= 1500)[0]}
out["stratified"] = {k: {"n_targets": int(len(ix)), "cv_tuned": block(g_cv, "cv", ix),
                         "implied": block(g_imp, "imp", ix)} for k, ix in strata.items() if len(ix)}
json.dump(out, open(HERE / "summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
