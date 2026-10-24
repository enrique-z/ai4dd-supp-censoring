#!/usr/bin/env python3
"""Post-hoc S4 optimiser / regularisation sensitivity for P3 (ChEMBL censoring).

Self-contained: imports nothing from the paper directory. Everything except the
fit function is a verbatim copy of analysis/run_censoring.py::s4_downstream
(SQL incl. relation filter, target selection, per-compound aggregation, Morgan
r=2/1024, split, seeds, matched subsample rng order).

Subcommands
  extract            SQL + featurise all eligible targets -> cache/<tid>.npz
  gate  --tids ...   ORIGINAL fit (verbatim) on cached data; compare to raw JSON
  sweep --lam L      CORRECTED fit, all 4 arms, all cached targets -> s4_lam_<L>.json
  orig_full          ORIGINAL fit on all targets with sigma logging -> s4_orig_full.json
  gradcheck          scipy.optimize.check_grad on small random problems
  summarise          summary JSON over all available s4_*.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import erf, log_ndtr

WORK = Path(__file__).resolve().parent
CACHE = WORK / "cache"
RAW = WORK.parent / "evidence" / "censoring_raw.json"
DB = os.environ.get("CHEMBL_DB", "./data/chembl_36/chembl_36_sqlite/chembl_36.db")


def pactivity(nm: float) -> float:
    return 9.0 - math.log10(nm)


def norm_pdf(z):
    return np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def norm_cdf(z):
    return 0.5 * (1.0 + erf(z / math.sqrt(2.0)))


# ------------------------------------------------------------ ORIGINAL (verbatim)
def tobit_fit_orig(X, y, is_right, n_iter=300, lr=0.05, seed=0, return_info=False):
    n, d = X.shape
    obs = ~is_right
    cen = is_right
    Xo = sparse.csr_matrix(X[obs]) if obs.any() else None
    XoT = sparse.csr_matrix(X[obs].T) if obs.any() else None
    Xc = sparse.csr_matrix(X[cen]) if cen.any() else None
    XcT = sparse.csr_matrix(X[cen].T) if cen.any() else None
    yo, yc = y[obs], y[cen]
    lam = 1e-3

    def nll_and_grad(theta):
        w = theta[:d]
        b = theta[d]
        log_sigma = float(np.clip(theta[d + 1], -5.0, 5.0))
        sigma = math.exp(log_sigma)
        f = 0.0
        gw = np.zeros(d)
        gb = 0.0
        gs = 0.0
        if Xo is not None:
            r = (yo - (Xo @ w + b)) / sigma
            f += float(0.5 * np.sum(r * r) + len(yo) * log_sigma)
            gw += -(XoT @ r) / sigma
            gb += -np.sum(r) / sigma
            gs += np.sum(1.0 - r * r)
        if Xc is not None:
            zc = (yc - (Xc @ w + b)) / sigma
            cdf = np.clip(norm_cdf(zc), 1e-12, 1.0 - 1e-12)
            f += float(-np.sum(np.log(cdf)))
            h = norm_pdf(zc) / cdf
            gw += (XcT @ h) / sigma
            gb += np.sum(h) / sigma
            gs += np.sum(h * zc)
        f = f / n + 0.5 * lam * float(w @ w)
        g = np.empty(d + 2)
        g[:d] = gw / n + lam * w
        g[d] = gb / n
        g[d + 1] = gs / n
        return f, g

    theta0 = np.zeros(d + 2)
    theta0[d] = float(np.mean(y))
    theta0[d + 1] = math.log(max(np.std(y), 1e-2))
    res = minimize(nll_and_grad, theta0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "maxfun": 4000, "ftol": 1e-10,
                            "gtol": 1e-7})
    w, b = res.x[:d], float(res.x[d])
    if return_info:
        ls_raw = float(res.x[d + 1])
        ls = float(np.clip(ls_raw, -5.0, 5.0))
        info = {"converged": bool(res.success), "n_iter": int(res.nit),
                "grad_norm": float(np.max(np.abs(res.jac))),
                "log_sigma_raw": ls_raw, "log_sigma": ls,
                "sigma_at_bound": bool(ls_raw <= -5.0 or ls_raw >= 5.0),
                "message": str(res.message)}
        if cen.any():
            zc = (yc - (Xc @ w + b)) / math.exp(ls)
            info["n_cens_z_lt_m7"] = int((zc < -7.0345).sum())
        return w, b, info
    return w, b


# ------------------------------------------------------------ CORRECTED
def make_objective(X, y, is_right, lam):
    n, d = X.shape
    obs = ~is_right
    cen = is_right
    Xo = sparse.csr_matrix(X[obs]) if obs.any() else None
    XoT = sparse.csr_matrix(X[obs].T) if obs.any() else None
    Xc = sparse.csr_matrix(X[cen]) if cen.any() else None
    XcT = sparse.csr_matrix(X[cen].T) if cen.any() else None
    yo, yc = y[obs], y[cen]
    half_log_2pi = 0.5 * math.log(2 * math.pi)

    def nll_and_grad(theta):
        w = theta[:d]
        b = theta[d]
        log_sigma = float(theta[d + 1])          # bounded by L-BFGS-B, no clip
        sigma = math.exp(log_sigma)
        f = 0.0
        gw = np.zeros(d)
        gb = 0.0
        gs = 0.0
        if Xo is not None:
            r = (yo - (Xo @ w + b)) / sigma
            f += float(0.5 * np.sum(r * r) + len(yo) * log_sigma)
            gw += -(XoT @ r) / sigma
            gb += -np.sum(r) / sigma
            gs += np.sum(1.0 - r * r)
        if Xc is not None:
            zc = (yc - (Xc @ w + b)) / sigma
            lcdf = log_ndtr(zc)
            f += float(-np.sum(lcdf))
            h = np.exp(-0.5 * zc * zc - half_log_2pi - lcdf)   # stable Mills ratio
            gw += (XcT @ h) / sigma
            gb += np.sum(h) / sigma
            gs += np.sum(h * zc)
        f = f / n + 0.5 * lam * float(w @ w)
        g = np.empty(d + 2)
        g[:d] = gw / n + lam * w
        g[d] = gb / n
        g[d + 1] = gs / n
        return f, g

    return nll_and_grad


def tobit_fit_corr(X, y, is_right, lam):
    n, d = X.shape
    fun = make_objective(X, y, is_right, lam)
    theta0 = np.zeros(d + 2)
    theta0[d] = float(np.mean(y))
    theta0[d + 1] = float(np.clip(math.log(max(np.std(y), 1e-2)), -5.0, 5.0))
    bounds = [(None, None)] * (d + 1) + [(-5.0, 5.0)]
    res = minimize(fun, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 2000, "maxfun": 4000, "ftol": 1e-10,
                            "gtol": 1e-7})
    w, b, ls = res.x[:d], float(res.x[d]), float(res.x[d + 1])
    info = {"converged": bool(res.success), "n_iter": int(res.nit),
            "grad_norm": float(np.max(np.abs(res.jac[:d + 1]))),
            "log_sigma": ls,
            "sigma_at_bound": bool(ls <= -5.0 + 1e-9 or ls >= 5.0 - 1e-9),
            "message": str(res.message)}
    if is_right.any():
        zc = (y[is_right] - (X[is_right] @ w + b)) / math.exp(ls)
        info["n_cens_z_lt_m7"] = int((zc < -7.0345).sum())
    return w, b, info


# ------------------------------------------------------------ data (verbatim SQL)
TGT_SQL = """select a.tid, count(*) c,
                      sum(case when act.standard_relation in ('>','>=','>>') then 1 else 0 end) r
               from activities act
               join assays a on act.assay_id = a.assay_id
               join target_dictionary td on a.tid = td.tid
               where act.standard_type='IC50' and act.standard_units='nM'
                 and act.standard_value > 0
                 and td.target_type = 'SINGLE PROTEIN'
               group by a.tid having c > 2000 order by c desc"""
ROW_SQL = """select cs.canonical_smiles, act.standard_value, act.standard_relation,
                          td.pref_name
                   from activities act
                   join assays a on act.assay_id = a.assay_id
                   join target_dictionary td on a.tid = td.tid
                   join compound_structures cs
                     on act.molregno = cs.molregno
                   where a.tid = ? and act.standard_type='IC50'
                     and act.standard_units='nM' and act.standard_value > 0
                     and act.standard_relation in ('=','>','>=','>>')"""


def cmd_extract(args):
    from rdkit import Chem, DataStructs, RDLogger, rdBase
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    if not Path(DB).is_file():
        print("FATAL: SD volume / db missing", file=sys.stderr)
        return 2
    CACHE.mkdir(exist_ok=True)
    con = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True)
    cur = con.cursor()
    t0 = time.time()
    tgt_rows = list(cur.execute(TGT_SQL))
    eligible = [(t, c, r) for t, c, r in tgt_rows if r >= 200]
    print(f"[extract] {len(eligible)} eligible ({time.time()-t0:.0f}s)", flush=True)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    kept = []
    for i_t, (tid, n_tot, n_right) in enumerate(eligible):
        if args.nshard > 1 and i_t % args.nshard != args.shard:
            continue
        out = CACHE / f"{tid}.npz"
        if out.exists():
            z = np.load(out, allow_pickle=False)
            if bool(z["kept"]):
                kept.append(int(tid))
            continue
        rows = list(cur.execute(ROW_SQL, (tid,)))
        tname = rows[0][3] if rows else str(tid)
        by_smi = {}
        for smi, val, rel, _ in rows:
            dd = by_smi.setdefault(smi, {"exact": [], "cens": []})
            dd["exact" if rel == "=" else "cens"].append(pactivity(val))
        X, y, isr = [], [], []
        for smi, dd in by_smi.items():
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            arr = np.zeros(1024, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m), arr)
            X.append(arr)
            if dd["exact"]:
                y.append(float(np.median(dd["exact"]))); isr.append(False)
            else:
                y.append(float(np.median(dd["cens"]))); isr.append(True)
        X = np.asarray(X, np.uint8).reshape(-1, 1024)
        y = np.asarray(y, float)
        isr = np.asarray(isr, bool)
        ok = not (len(X) < 300 or isr.sum() < 30 or (~isr).sum() < 150)
        np.savez_compressed(out, X=np.packbits(X, axis=1), y=y, isr=isr,
                            name=np.array(tname), kept=np.array(ok),
                            n_tot=np.array(n_tot), n_right_sql=np.array(n_right))
        if ok:
            kept.append(int(tid))
        print(f"[extract] tid={tid} n={len(X)} cens={int(isr.sum())} kept={ok} "
              f"({time.time()-t0:.0f}s)", flush=True)
    if args.nshard > 1:
        return 0
    meta = {"eligible": [int(t) for t, _, _ in eligible], "kept": kept,
            "rdkit": rdBase.rdkitVersion, "numpy": np.__version__,
            "python": sys.version.split()[0], "sqlite": sqlite3.sqlite_version}
    (CACHE / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[extract] kept {len(kept)} in {time.time()-t0:.0f}s", flush=True)
    return 0


def load_target(tid):
    z = np.load(CACHE / f"{tid}.npz", allow_pickle=False)
    X = np.unpackbits(z["X"], axis=1)[:, :1024].astype(float)
    return X, z["y"], z["isr"], str(z["name"])


# ------------------------------------------------------------ per-target run
LAM_GRID = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0)
N_FOLDS = 5


def cv_choose_lam(X, y, isr, rows, cens_mask, perm_seed):
    """5-fold CV over LAM_GRID on one arm's TRAINING rows only.

    rows: indices (into X) of the arm's training rows; cens_mask: bool per row,
    True = enters the fit as a censored bound (Tobit arm only). Held-out rows
    are scored only if they are EXACT records (isr False); censored rows are
    never scored. Score = pooled RMSE over all held-out exact rows of the 5
    folds. Ties -> smallest lambda (first in grid)."""
    perm = np.random.default_rng(perm_seed).permutation(len(rows))
    fold = np.empty(len(rows), int)
    fold[perm] = np.arange(len(rows)) % N_FOLDS
    scores = []
    for lam in LAM_GRID:
        sse, cnt = 0.0, 0
        for k in range(N_FOLDS):
            trk, hok = fold != k, fold == k
            r_tr, r_ho = rows[trk], rows[hok]
            w, b, _ = tobit_fit_corr(X[r_tr], y[r_tr], cens_mask[trk], lam)
            ex = r_ho[~isr[r_ho]]
            res = X[ex] @ w + b - y[ex]
            sse += float(res @ res); cnt += len(ex)
        scores.append(math.sqrt(sse / cnt))
    j = int(np.argmin(scores))
    return LAM_GRID[j], scores


def run_target_tuned(tid):
    t0 = time.time()
    X, y, isr, tname = load_target(tid)
    per_seed = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        exact_idx = np.where(~isr)[0]
        rng.shuffle(exact_idx)
        n_te = max(50, int(0.25 * len(exact_idx)))
        te = exact_idx[:n_te]
        tr_exact = exact_idx[n_te:]
        tr_cens = np.where(isr)[0]
        tr_all = np.concatenate([tr_exact, tr_cens])
        tr_m = rng.choice(tr_all, size=len(tr_exact), replace=False)

        def rmse(w, b, idx):
            return float(np.sqrt(np.mean((X[idx] @ w + b - y[idx]) ** 2)))

        # inner-fold permutation seeds: one per distinct training-row set, so
        # naive and Tobit (both tr_all) share folds
        arms = (("naive", tr_all, np.zeros(len(tr_all), bool), 10_000 + seed),
                ("exact_only", tr_exact, np.zeros(len(tr_exact), bool), 20_000 + seed),
                ("censoring_aware", tr_all, np.isin(tr_all, tr_cens), 10_000 + seed),
                ("naive_matched", tr_m, np.zeros(len(tr_m), bool), 30_000 + seed))
        rec = {"seed": seed, "n_train_exact_only": int(len(tr_exact)),
               "n_train_all": int(len(tr_all))}
        for name, rows, cm, ps in arms:
            lam, scores = cv_choose_lam(X, y, isr, rows, cm, ps)
            w, b, info = tobit_fit_corr(X[rows], y[rows], cm, lam)
            info["lam_chosen"] = lam
            info["cv_rmse"] = [round(v, 6) for v in scores]
            rec[f"rmse_{name}"] = rmse(w, b, te)
            rec[f"info_{name}"] = info
        per_seed.append(rec)
    keys = ("rmse_naive", "rmse_exact_only", "rmse_censoring_aware", "rmse_naive_matched")
    agg = {k: round(float(np.mean([p[k] for p in per_seed])), 5) for k in keys}
    return {"tid": int(tid), "target": tname, "n_compounds": int(len(X)),
            "n_right_censored": int(isr.sum()),
            "censored_frac": round(float(isr.mean()), 4),
            "n_seeds": 10, **agg, "per_seed": per_seed,
            "seconds": round(time.time() - t0, 1)}


def run_target(job):
    tid, mode, lam = job
    if mode == "tuned":
        return run_target_tuned(tid)
    t0 = time.time()
    X, y, isr, tname = load_target(tid)
    if mode == "orig":
        fit = lambda Xa, ya, ra: tobit_fit_orig(Xa, ya, ra, return_info=True)
    else:
        fit = lambda Xa, ya, ra: tobit_fit_corr(Xa, ya, ra, lam)
    per_seed = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        exact_idx = np.where(~isr)[0]
        rng.shuffle(exact_idx)
        n_te = max(50, int(0.25 * len(exact_idx)))
        te = exact_idx[:n_te]
        tr_exact = exact_idx[n_te:]
        tr_cens = np.where(isr)[0]
        tr_all = np.concatenate([tr_exact, tr_cens])

        def rmse(w, b, idx):
            return float(np.sqrt(np.mean((X[idx] @ w + b - y[idx]) ** 2)))

        wn, bn, in_ = fit(X[tr_all], y[tr_all], np.zeros(len(tr_all), bool))
        we, be, ie_ = fit(X[tr_exact], y[tr_exact], np.zeros(len(tr_exact), bool))
        wc, bc, ic_ = fit(X[tr_all], y[tr_all], np.isin(tr_all, tr_cens))
        tr_m = rng.choice(tr_all, size=len(tr_exact), replace=False)
        wm, bm, im_ = fit(X[tr_m], y[tr_m], np.zeros(len(tr_m), bool))
        per_seed.append({
            "seed": seed,
            "rmse_naive": rmse(wn, bn, te),
            "rmse_exact_only": rmse(we, be, te),
            "rmse_censoring_aware": rmse(wc, bc, te),
            "rmse_naive_matched": rmse(wm, bm, te),
            "n_train_exact_only": int(len(tr_exact)),
            "n_train_all": int(len(tr_all)),
            "n_train_matched_censored": int(np.isin(tr_m, tr_cens).sum()),
            "info_naive": in_, "info_exact_only": ie_,
            "info_censoring_aware": ic_, "info_naive_matched": im_,
        })
    keys = ("rmse_naive", "rmse_exact_only", "rmse_censoring_aware", "rmse_naive_matched")
    agg = {k: round(float(np.mean([p[k] for p in per_seed])), 5) for k in keys}
    return {"tid": int(tid), "target": tname, "n_compounds": int(len(X)),
            "n_right_censored": int(isr.sum()),
            "censored_frac": round(float(isr.mean()), 4),
            "n_seeds": 10, **agg, "per_seed": per_seed,
            "seconds": round(time.time() - t0, 1)}


def run_pool(jobs, nproc, out_jsonl):
    import multiprocessing as mp
    done = {}
    if out_jsonl.exists():
        for line in out_jsonl.read_text().splitlines():
            r = json.loads(line)
            done[r["tid"]] = r
    todo = [j for j in jobs if j[0] not in done]
    print(f"[pool] {len(done)} done, {len(todo)} to do, nproc={nproc}", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(nproc, maxtasksperchild=4) as pool, out_jsonl.open("a") as fh:
        for i, r in enumerate(pool.imap_unordered(run_target, todo), 1):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            done[r["tid"]] = r
            print(f"[pool] {i}/{len(todo)} tid={r['tid']} {r['seconds']}s "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
    return done


def kept_tids():
    return json.loads((CACHE / "meta.json").read_text())["kept"]


def cmd_gate(args):
    raw = json.loads(RAW.read_text())["S4_downstream"]["targets"]
    pub = {t["tid"]: t for t in raw}
    meta = json.loads((CACHE / "meta.json").read_text())
    tids_pub = [t["tid"] for t in raw]
    print(f"[gate] kept tids match published (same order): {meta['kept'] == tids_pub}; "
          f"set equal: {set(meta['kept']) == set(tids_pub)}", flush=True)
    import multiprocessing as mp
    jobs = [(int(t), "orig", 1e-3) for t in args.tids]
    with mp.get_context("spawn").Pool(min(len(jobs), args.nproc)) as pool:
        res = pool.map(run_target, jobs)
    report = []
    worst = 0.0
    for r in res:
        p = pub[r["tid"]]
        assert p["n_compounds"] == r["n_compounds"], (p["n_compounds"], r["n_compounds"])
        for a, b in zip(r["per_seed"], p["per_seed"]):
            for k in ("rmse_naive", "rmse_exact_only", "rmse_censoring_aware",
                      "rmse_naive_matched"):
                worst = max(worst, abs(a[k] - b[k]))
            for k, ik in (("converged_naive", "info_naive"),
                          ("converged_exact_only", "info_exact_only"),
                          ("converged_censoring_aware", "info_censoring_aware"),
                          ("converged_naive_matched", "info_naive_matched")):
                if a[ik]["converged"] != b[k]:
                    print(f"[gate] conv flag mismatch tid={r['tid']} seed={a['seed']} {k}")
            assert a["n_train_exact_only"] == b["n_train_exact_only"]
        md = max(abs(a[k] - b[k]) for a, b in zip(r["per_seed"], p["per_seed"])
                 for k in ("rmse_naive", "rmse_exact_only", "rmse_censoring_aware",
                           "rmse_naive_matched"))
        report.append({"tid": r["tid"], "target": r["target"],
                       "n_exact_train": r["per_seed"][0]["n_train_exact_only"],
                       "max_abs_diff": md})
        print(f"[gate] tid={r['tid']} n_ex_tr={r['per_seed'][0]['n_train_exact_only']} "
              f"max|diff|={md:.3e} ({r['seconds']}s)", flush=True)
    out = {"targets": report, "worst_abs_diff": worst, "pass": worst < 1e-6,
           "kept_matches_published_order": meta["kept"] == tids_pub, "env": meta,
           "fits": res}
    (WORK / "gate_result.json").write_text(json.dumps(out, indent=1))
    print(f"[gate] worst={worst:.3e} PASS={worst < 1e-6}", flush=True)
    return 0 if worst < 1e-6 else 1


def cmd_sweep(args):
    tids = kept_tids()
    tag = "orig_full" if args.orig else ("tuned" if args.tuned else f"lam_{args.lam:g}")
    mode = "orig" if args.orig else ("tuned" if args.tuned else "corr")
    if args.tids:
        tids = [t for t in tids if t in set(args.tids)]
        tag += "_subset"
    jl = WORK / f"s4_{tag}.jsonl"
    done = run_pool([(t, mode, args.lam) for t in tids], args.nproc, jl)
    res = [done[t] for t in tids]
    (WORK / f"s4_{tag}.json").write_text(json.dumps(
        {"lam": args.lam, "mode": mode, "targets": res}, indent=0))
    print(f"[sweep] wrote s4_{tag}.json", flush=True)
    return 0


def cmd_gradcheck(args):
    from scipy.optimize import check_grad
    rng = np.random.default_rng(0)
    out = []
    for trial in range(6):
        n, d = 80, 30
        X = (rng.random((n, d)) < 0.2).astype(float)
        y = rng.normal(6, 1, n)
        isr = rng.random(n) < 0.4
        lam = [1e-3, 1e-1, 1.0][trial % 3]
        fun = make_objective(X, y, isr, lam)
        f = lambda t: fun(t)[0]
        g = lambda t: fun(t)[1]
        for scale in (0.1, 2.0, 8.0):  # 8.0 pushes censored z far into the tail
            th = np.concatenate([rng.normal(0, scale, d), [rng.normal(6, 1)],
                                 [rng.uniform(-3, 1)]])
            err = check_grad(f, g, th, epsilon=1e-7)
            rel = err / max(1e-12, np.linalg.norm(g(th)))
            Xc = X[isr]; zc = (y[isr] - (Xc @ th[:d] + th[d])) / math.exp(th[d + 1])
            out.append({"trial": trial, "lam": lam, "scale": scale, "abs_err": err,
                        "rel_err": rel, "min_z_cens": float(zc.min())})
            print(f"[gradcheck] lam={lam} scale={scale} abs={err:.2e} rel={rel:.2e} "
                  f"minz={zc.min():.1f}")
    (WORK / "gradcheck.json").write_text(json.dumps(out, indent=1))
    return 0


# ------------------------------------------------------------ summary
def boot_targets(vals, n_boot=10000, seed=20260904):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    rng = np.random.default_rng(seed)
    draws = np.mean(v[rng.integers(0, len(v), size=(n_boot, len(v)))], axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


CONTRASTS = (("naive_minus_exact", "rmse_naive", "rmse_exact_only"),
             ("matched_minus_exact", "rmse_naive_matched", "rmse_exact_only"),
             ("tobit_minus_exact", "rmse_censoring_aware", "rmse_exact_only"),
             ("tobit_minus_naive", "rmse_censoring_aware", "rmse_naive"))
ARMS = (("naive", "info_naive"), ("exact", "info_exact_only"),
        ("tobit", "info_censoring_aware"), ("matched", "info_naive_matched"))


def summarise_targets(tg):
    from scipy import stats
    out = {"n_targets": len(tg)}
    cf = np.array([x["censored_frac"] for x in tg], float)
    nex = np.array([x["per_seed"][0]["n_train_exact_only"] for x in tg])
    strata = {"lt1024": nex < 1024, "1024_1500": (nex >= 1024) & (nex < 1500),
              "ge1500": nex >= 1500}
    for name, a, b in CONTRASTS:
        dif = np.array([x[a] - x[b] for x in tg], float)
        lo, hi = boot_targets(dif)
        e = {"mean": round(float(dif.mean()), 4), "ci": [round(lo, 4), round(hi, 4)],
             "median": round(float(np.median(dif)), 4),
             "n_worse": int((dif > 0).sum()), "n_better": int((dif < 0).sum()),
             "strata": {}}
        for s, m in strata.items():
            if m.sum() >= 2:
                slo, shi = boot_targets(dif[m])
                e["strata"][s] = {"n": int(m.sum()), "mean": round(float(dif[m].mean()), 4),
                                  "ci": [round(slo, 4), round(shi, 4)],
                                  "n_worse": int((dif[m] > 0).sum())}
        out[name] = e
    for name, key in (("p3_rho", "naive_minus_exact"), ("p3_rho_matched", "matched_minus_exact")):
        a, b = {"naive_minus_exact": ("rmse_naive", "rmse_exact_only"),
                "matched_minus_exact": ("rmse_naive_matched", "rmse_exact_only")}[key]
        dif = np.array([x[a] - x[b] for x in tg], float)
        rho, p = stats.spearmanr(cf, dif)
        out[name] = {"rho": round(float(rho), 3), "p": float(f"{p:.3g}")}
    for k in ("rmse_naive", "rmse_exact_only", "rmse_censoring_aware", "rmse_naive_matched"):
        out[f"mean_{k}"] = round(float(np.mean([x[k] for x in tg])), 4)
    # per-arm fit diagnostics (if present)
    if "info_naive" in tg[0]["per_seed"][0]:
        diag = {}
        for arm, ik in ARMS:
            infos = [ps[ik] for x in tg for ps in x["per_seed"]]
            ls = np.array([i["log_sigma"] for i in infos])
            diag[arm] = {
                "n_fits": len(infos),
                "converged_pct": round(100 * np.mean([i["converged"] for i in infos]), 2),
                "sigma_at_lower_bound_pct": round(100 * float(np.mean(ls <= -5 + 1e-9)), 2),
                "sigma_at_any_bound_pct": round(100 * np.mean([i["sigma_at_bound"] for i in infos]), 2),
                "median_sigma": round(float(np.exp(np.median(ls))), 4),
                "median_n_iter": float(np.median([i["n_iter"] for i in infos])),
                "fits_with_cens_z_lt_m7": int(sum(1 for i in infos if i.get("n_cens_z_lt_m7", 0) > 0)),
                "sigma_at_lower_by_stratum_pct": {
                    s: round(100 * float(np.mean([ps[ik]["log_sigma"] <= -5 + 1e-9
                                                  for x, mm in zip(tg, m) if mm
                                                  for ps in x["per_seed"]])), 2)
                    for s, m in strata.items()},
            }
        out["fit_diagnostics"] = diag
    return out


def cmd_summarise(args):
    raw = json.loads(RAW.read_text())["S4_downstream"]["targets"]
    summ = {"bootstrap": "10000 resamples over targets, np.random.default_rng(20260904), "
                         "percentile 2.5/97.5 (as make_ledger.py boot_targets)",
            "strata_by": "n_train_exact_only (seed 0; identical across seeds)",
            "published_original": summarise_targets(raw)}
    for f in sorted(WORK.glob("s4_*.json")):
        if "subset" in f.stem or f.stem.startswith("s4_sensitivity"):
            continue
        d = json.loads(f.read_text())
        tg = d["targets"]
        assert [t["tid"] for t in tg] == [t["tid"] for t in raw]
        summ[f.stem] = summarise_targets(tg)
        summ[f.stem]["lam"] = d["lam"]; summ[f.stem]["mode"] = d["mode"]
        if d["mode"] == "tuned":
            summ[f.stem]["lam"] = "5-fold CV per target x seed x arm"
            lamd = {}
            for arm, ik in ARMS:
                lc = np.array([ps[ik]["lam_chosen"] for x in tg for ps in x["per_seed"]])
                vals, cnts = np.unique(lc, return_counts=True)
                lamd[arm] = {"median": float(np.median(lc)), "min": float(lc.min()),
                             "max": float(lc.max()),
                             "at_low_edge_pct": round(100 * float(np.mean(lc == LAM_GRID[0])), 2),
                             "at_high_edge_pct": round(100 * float(np.mean(lc == LAM_GRID[-1])), 2),
                             "counts": {f"{v:g}": int(c) for v, c in zip(vals, cnts)}}
            summ[f.stem]["lam_chosen_distribution"] = lamd
    (WORK / "s4_sensitivity_summary.json").write_text(json.dumps(summ, indent=1))
    for k, v in summ.items():
        if isinstance(v, dict) and "naive_minus_exact" in v:
            print(k, {c: (v[c]["mean"], v[c]["ci"], v[c]["n_worse"]) for c, _, _ in CONTRASTS},
                  v["p3_rho"], v["p3_rho_matched"])
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--tids", type=int, nargs="*", default=[])
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--orig", action="store_true")
    ap.add_argument("--tuned", action="store_true")
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()
    return {"extract": cmd_extract, "gate": cmd_gate, "sweep": cmd_sweep,
            "gradcheck": cmd_gradcheck, "summarise": cmd_summarise}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
