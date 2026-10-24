"""Independent S4 re-analysis with plain ridge regression (intercept unpenalised).

For each target x seed: reproduce the original split / matched-subsample rng
draws, then fit three arms (naive, exact_only, matched) with
  min_w,b ||y - Xw - b||^2 + alpha ||w||^2
via an eigendecomposition of the centred Gram matrix (exact for every alpha).

alpha choices recorded per arm:
  * cv     : 5-fold CV on that arm's TRAINING rows only (test never touched)
  * implied: the self-consistent alpha of the original objective,
             (1/n)[0.5 sum r^2/s^2 + n log s] + 0.5 lam ||w||^2, lam = 1e-3,
             whose stationarity conditions are alpha = n lam s^2, s^2 = RSS/n,
             i.e. alpha = lam * RSS(alpha). Solved by root-finding.
  * grid   : test RMSE at every grid alpha (for fixed-alpha reporting)
Naive arm also gets 'cvx': CV where validation folds are scored on exact rows only.

Usage: ridge_cv.py [n_seeds] [n_workers] [tid ...]
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json, sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
GRID = 10.0 ** np.arange(-2, 4.01, 0.25)      # 25 values, 1e-2 .. 1e4
LAM = 1e-3


class Path_:
    """Ridge solution path from raw sufficient statistics."""
    def __init__(self, G, sx, sy, syy, n):
        mu = sx / n
        ym = sy / n
        Gc = G - n * np.outer(mu, mu)
        xy = None
        self.mu, self.ym, self.n = mu, ym, n
        s, V = np.linalg.eigh(Gc)
        self.s = np.clip(s, 0.0, None)
        self.V = V
        self.tss = syy - n * ym * ym

    def set_xy(self, sxy):
        xyc = sxy - self.n * self.mu * self.ym
        self.z = self.V.T @ xyc

    def coefs(self, alphas):                     # 1024 x A in eigen-basis
        return self.z[:, None] / (self.s[:, None] + np.asarray(alphas)[None, :])

    def predict(self, Xq, alphas):
        P = (Xq - self.mu) @ self.V
        return P @ self.coefs(alphas) + self.ym

    def rss(self, a):
        k = self.s > 1e-9 * self.s.max()          # null-space components carry z = 0
        s, z = self.s[k], self.z[k]
        return float(self.tss - np.sum(z ** 2 * (s + 2 * a) / (s + a) ** 2))

    def implied_alpha(self, lam=LAM):
        g = lambda a: a - lam * self.rss(a)
        hi = lam * self.tss
        a_clip = self.n * lam * np.exp(-10.0)    # original clips log sigma >= -5
        grid = np.logspace(-12, np.log10(hi), 500)
        gv = np.array([g(a) for a in grid])
        roots = [i for i in range(len(grid) - 1) if gv[i] < 0 <= gv[i + 1]]
        if not roots:
            # no interior root: optimum drives sigma to the original's clip
            # (log sigma >= -5), so alpha = n * lam * exp(-10)
            return float(a_clip), 0
        i = roots[-1]                            # largest root (L-BFGS starts at large sigma)
        a = float(brentq(g, grid[i], grid[i + 1], xtol=1e-14))
        return max(a, float(a_clip)), len(roots)


def stats(X, y):
    return X.T @ X, X.sum(0), float(y.sum()), float(y @ y), X.T @ y


def fit_full(X, y):
    G, sx, sy, syy, sxy = stats(X, y)
    p = Path_(G, sx, sy, syy, len(y)); p.set_xy(sxy)
    return p


def cv_curve(X, y, w_val, seed, k=5):
    """5-fold CV MSE per GRID alpha; w_val = per-row weight for validation scoring."""
    n = len(y)
    perm = np.random.default_rng(10_000 + seed).permutation(n)
    folds = np.array_split(perm, k)
    G, sx, sy, syy, sxy = stats(X, y)
    sse = np.zeros(len(GRID)); wsum = 0.0
    for f in folds:
        Xf, yf = X[f], y[f]
        p = Path_(G - Xf.T @ Xf, sx - Xf.sum(0), sy - yf.sum(), syy - yf @ yf, n - len(f))
        p.set_xy(sxy - Xf.T @ yf)
        wf = w_val[f]
        if wf.sum() == 0:
            continue
        pred = p.predict(Xf, GRID)
        sse += (wf[:, None] * (pred - yf[:, None]) ** 2).sum(0)
        wsum += wf.sum()
    return sse / wsum


def rmse_at(p, Xte, yte, alphas):
    pred = p.predict(Xte, alphas)
    return np.sqrt(np.mean((pred - yte[:, None]) ** 2, axis=0))


def run_target(args):
    tid, n_seeds = args
    d = np.load(CACHE / f"t{tid}.npz")
    X = d["X"].astype(np.float64); y = d["y"]; isr = d["isr"]
    out = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        exact_idx = np.where(~isr)[0]
        rng.shuffle(exact_idx)
        n_te = max(50, int(0.25 * len(exact_idx)))
        te = exact_idx[:n_te]
        tr_exact = exact_idx[n_te:]
        tr_cens = np.where(isr)[0]
        tr_all = np.concatenate([tr_exact, tr_cens])
        tr_m = rng.choice(tr_all, size=len(tr_exact), replace=False)
        Xte, yte = X[te], y[te]
        rec = {"seed": seed, "n_train_exact_only": int(len(tr_exact)),
               "n_train_all": int(len(tr_all)), "n_test": int(n_te),
               "n_cens_in_matched": int(isr[tr_m].sum())}
        for arm, idx in (("naive", tr_all), ("exact_only", tr_exact), ("matched", tr_m)):
            Xa, ya = X[idx], y[idx]
            p = fit_full(Xa, ya)
            grid_rmse = rmse_at(p, Xte, yte, GRID)
            cv = cv_curve(Xa, ya, np.ones(len(idx)), seed)
            j = int(np.argmin(cv))
            a_imp, nroots = p.implied_alpha()
            r_imp = float(rmse_at(p, Xte, yte, [a_imp])[0])
            rec[arm] = {"alpha_cv": float(GRID[j]), "cv_edge": j in (0, len(GRID) - 1),
                        "rmse_cv": float(grid_rmse[j]),
                        "alpha_implied": a_imp, "n_roots": nroots,
                        "sigma_implied": float(np.sqrt(max(p.rss(a_imp), 0) / len(idx))),
                        "rmse_implied": r_imp,
                        "rmse_grid": [float(v) for v in grid_rmse]}
            if arm in ("naive", "matched"):
                cvx = cv_curve(Xa, ya, (~isr[idx]).astype(float), seed)
                jx = int(np.argmin(cvx))
                rec[arm]["alpha_cvx"] = float(GRID[jx])
                rec[arm]["rmse_cvx"] = float(grid_rmse[jx])
        out.append(rec)
    return {"tid": int(tid), "n_compounds": int(len(y)), "n_cens": int(isr.sum()),
            "per_seed": out}


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    tids = [int(t) for t in sys.argv[3:]] or json.load(open(CACHE / "kept.json"))
    t0 = time.time()
    res = []
    with Pool(nw) as pool:
        for r in pool.imap_unordered(run_target, [(t, n_seeds) for t in tids]):
            res.append(r)
            print(f"[{len(res)}/{len(tids)}] tid={r['tid']} {time.time()-t0:.0f}s", flush=True)
    tag = f"s{n_seeds}_n{len(tids)}"
    json.dump({"grid": GRID.tolist(), "lam": LAM, "n_seeds": n_seeds, "results": res},
              open(HERE / f"ridge_results_{tag}.json", "w"))
    print(f"done {time.time()-t0:.0f}s -> ridge_results_{tag}.json")
