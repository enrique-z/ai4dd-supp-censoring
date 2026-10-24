"""Summarise the original-parameterisation lambda sweep."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
D = json.load(open(sys.argv[1])); R = sorted(D["results"], key=lambda r: r["tid"])
cf = np.array([r["censored_frac"] for r in R])
rng = np.random.default_rng(2024)
B = rng.integers(0, len(R), (10_000, len(R)))
out = {"n_targets": len(R), "n_seeds": len(R[0]["per_seed"])}
for lam in map(str, D["lams"]):
    pt = lambda a, b: np.array([np.mean([p[lam][a] - p[lam][b] for p in r["per_seed"]]) for r in R])
    L = {}
    for name, a, b in (("naive_minus_exact", "naive", "exact_only"),
                       ("matched_minus_exact", "matched", "exact_only"),
                       ("tobit_minus_exact", "tobit", "exact_only"),
                       ("tobit_minus_naive", "tobit", "naive")):
        v = pt(a, b); bs = v[B].mean(1)
        L[name] = {"mean": round(float(v.mean()), 5),
                   "ci95": [round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)],
                   "n_first_worse": int((v > 0).sum())}
        if name in ("naive_minus_exact", "matched_minus_exact"):
            rho, pv = spearmanr(v, cf)
            L[name]["spearman_vs_censored_frac"] = round(float(rho), 4)
            L[name]["spearman_p"] = float(pv)
    fits = [p[lam] for r in R for p in r["per_seed"]]
    L["tobit_converged"] = int(sum(f["tobit_converged"] for f in fits))
    L["tobit_gradmax_max"] = float(max(f["tobit_gradmax"] for f in fits))
    L["logsig_at_lower_bound"] = {a: int(sum(f[a + "_logsig"] <= -5 + 1e-6 for f in fits))
                                  for a in ("naive", "exact_only", "matched", "tobit")}
    L["n_fits"] = len(fits)
    out[lam] = L
json.dump(out, open(HERE / "sweep_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
