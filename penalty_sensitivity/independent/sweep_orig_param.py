"""Fixed-lambda sweep in the ORIGINAL parameterisation, solved correctly.

Objective (per arm), theta = (w, b, t = log sigma), t bounded to [-5, 5]:
  f = (1/n) [ sum_obs (0.5 r_i^2 + t) - sum_cen log Phi((y_i - mu_i)/sigma) ] + 0.5 lam ||w||^2
  r_i = (y_i - mu_i)/sigma, mu = Xw + b, intercept unpenalised.

Gaussian arms (naive, exact_only, matched: no censored term) are solved by an
exact PROFILE over t: for fixed sigma the w-problem is ridge with
alpha = n lam sigma^2 and minimum value (TSS - z'(S+alpha)^-1 z)/(2 n sigma^2) + t
(eigen-path), so f(t) is a 1-D function minimised globally (grid + bounded refine).
Tobit arm: L-BFGS-B with log_ndtr, analytic gradient, t bounded, warm-started
at the naive-arm profile solution for the same lambda.

Usage: sweep_orig_param.py [n_seeds] [n_workers] [tid ...]
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json, math, sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy import sparse
from scipy.optimize import minimize, minimize_scalar
from scipy.special import log_ndtr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ridge_cv import fit_full, CACHE  # eigen-path ridge (validated vs sklearn)

LAMS = (1e-3, 1e-2, 1e-1, 1.0)
LOG2PI = math.log(2 * math.pi)


def profile_solve(p, lam):
    """Global min over t in [-5,5] of the profiled Gaussian objective."""
    n = p.n
    k = p.s > 1e-9 * p.s.max()
    s, z = p.s[k], p.z[k]

    def f(t):
        a = n * lam * math.exp(2 * t)
        return (p.tss - np.sum(z * z / (s + a))) / (2 * n * math.exp(2 * t)) + t

    grid = np.linspace(-5, 5, 401)
    fv = np.array([f(t) for t in grid])
    i = int(np.argmin(fv))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    r = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
    t = float(r.x) if r.fun <= fv[i] else float(grid[i])
    return t, n * lam * math.exp(2 * t)


def coefs_raw(p, a):
    """w (original feature basis) and b for ridge alpha a."""
    w = p.V @ (p.z / (p.s + a))
    b = p.ym - p.mu @ w
    return w, b


def tobit_fit(Xs, XsT, y, cen, lam, theta0):
    n, d = Xs.shape
    obs = ~cen
    n_obs = int(obs.sum())

    def fg(th):
        w, b, t = th[:d], th[d], th[d + 1]
        sig = math.exp(t)
        mu = Xs @ w + b
        zz = (y - mu) / sig
        zo, zc = zz[obs], zz[cen]
        lc = log_ndtr(zc)
        f = (0.5 * zo @ zo + n_obs * t - lc.sum()) / n + 0.5 * lam * (w @ w)
        h = np.exp(-0.5 * zc * zc - 0.5 * LOG2PI - lc)        # phi/Phi, stable
        dmu = np.empty(n)                                      # d f*n / d mu
        dmu[obs] = -zo / sig
        dmu[cen] = h / sig
        gt = (np.sum(1.0 - zo * zo) + np.sum(h * zc)) / n
        g = np.empty(d + 2)
        g[:d] = (XsT @ dmu) / n + lam * w
        g[d] = dmu.sum() / n
        g[d + 1] = gt
        return f, g

    bounds = [(None, None)] * (d + 1) + [(-5.0, 5.0)]
    r = minimize(fg, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                 options={"maxiter": 20000, "maxfun": 40000, "ftol": 1e-13, "gtol": 1e-8})
    g = fg(r.x)[1]
    gt = 0.0 if (r.x[d + 1] <= -5 + 1e-9 or r.x[d + 1] >= 5 - 1e-9) else abs(g[d + 1])
    return r.x[:d], float(r.x[d]), float(r.x[d + 1]), bool(r.success), float(max(np.max(np.abs(g[:d + 1])), gt))


def run_target(args):
    tid, n_seeds = args
    D = np.load(CACHE / f"t{tid}.npz")
    X = D["X"].astype(np.float64); y = D["y"]; isr = D["isr"]
    out = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        exact_idx = np.where(~isr)[0]
        rng.shuffle(exact_idx)
        n_te = max(50, int(0.25 * len(exact_idx)))
        te = exact_idx[:n_te]; tr_exact = exact_idx[n_te:]
        tr_cens = np.where(isr)[0]
        tr_all = np.concatenate([tr_exact, tr_cens])
        tr_m = rng.choice(tr_all, size=len(tr_exact), replace=False)
        Xte, yte = X[te], y[te]
        rm = lambda w, b: float(np.sqrt(np.mean((Xte @ w + b - yte) ** 2)))
        paths = {arm: fit_full(X[idx], y[idx]) for arm, idx in
                 (("naive", tr_all), ("exact_only", tr_exact), ("matched", tr_m))}
        Xs = sparse.csr_matrix(X[tr_all]); XsT = sparse.csr_matrix(X[tr_all].T)
        cen = isr[tr_all]
        rec = {"seed": seed}
        for lam in LAMS:
            L = {}
            for arm, p in paths.items():
                t, a = profile_solve(p, lam)
                w, b = coefs_raw(p, a)
                L[arm] = rm(w, b); L[arm + "_logsig"] = t
                if arm == "naive":
                    th0 = np.concatenate([w, [b, t]])
            wt, bt, tt, conv, gn = tobit_fit(Xs, XsT, y[tr_all], cen, lam, th0)
            L["tobit"] = rm(wt, bt); L["tobit_logsig"] = tt
            L["tobit_converged"] = conv; L["tobit_gradmax"] = gn
            rec[str(lam)] = L
        out.append(rec)
    return {"tid": int(tid), "censored_frac": float(isr.mean()), "per_seed": out}


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 13
    tids = [int(t) for t in sys.argv[3:]] or json.load(open(CACHE / "kept.json"))
    t0 = time.time(); res = []
    with Pool(nw) as pool:
        for r in pool.imap_unordered(run_target, [(t, n_seeds) for t in tids]):
            res.append(r)
            print(f"[{len(res)}/{len(tids)}] tid={r['tid']} {time.time()-t0:.0f}s", flush=True)
    tag = f"s{n_seeds}_n{len(tids)}"
    json.dump({"lams": LAMS, "results": res}, open(HERE / f"sweep_results_{tag}.json", "w"))
    print(f"done {time.time()-t0:.0f}s -> sweep_results_{tag}.json")
