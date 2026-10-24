"""Independent re-extraction of the S4 per-target datasets (ChEMBL 36, read-only).

Re-implemented from reading run_censoring.py::s4_downstream. Same SQL (so the
SQLite row order, hence dict insertion order, hence index order, matches),
same per-compound aggregation, same Morgan r=2 / 1024-bit featurisation,
same per-target minimums. Writes one npz per kept target into cache/.
"""
import json, math, os, sqlite3, sys, time
from pathlib import Path
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
DB = os.environ.get("CHEMBL_DB", "./data/chembl_36/chembl_36_sqlite/chembl_36.db")
OUT = Path(__file__).resolve().parent / "cache"
OUT.mkdir(exist_ok=True)

con = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True)
cur = con.cursor()
t0 = time.time()
tsel = OUT / "targets.json"
if tsel.exists():
    tgt_rows = json.load(open(tsel))
else:
    tgt_rows = list(cur.execute(
        """select a.tid, count(*) c,
                  sum(case when act.standard_relation in ('>','>=','>>') then 1 else 0 end) r
           from activities act
           join assays a on act.assay_id = a.assay_id
           join target_dictionary td on a.tid = td.tid
           where act.standard_type='IC50' and act.standard_units='nM'
             and act.standard_value > 0
             and td.target_type = 'SINGLE PROTEIN'
           group by a.tid having c > 2000 order by c desc"""))
    json.dump(tgt_rows, open(tsel, "w"))
eligible = [(t, c, r) for t, c, r in tgt_rows if r >= 200]
print(f"{len(eligible)} eligible targets ({time.time()-t0:.0f}s)", flush=True)

def one(tid):
    f = OUT / f"t{tid}.npz"
    if f.exists():
        return tid
    cur = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True).cursor()
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    rows = list(cur.execute(
        """select cs.canonical_smiles, act.standard_value, act.standard_relation,
                  td.pref_name
           from activities act
           join assays a on act.assay_id = a.assay_id
           join target_dictionary td on a.tid = td.tid
           join compound_structures cs
             on act.molregno = cs.molregno
           where a.tid = ? and act.standard_type='IC50'
             and act.standard_units='nM' and act.standard_value > 0
             and act.standard_relation in ('=','>','>=','>>')""", (tid,)))
    by = {}
    for smi, val, rel, _ in rows:
        d = by.setdefault(smi, ([], []))
        (d[0] if rel == "=" else d[1]).append(9.0 - math.log10(val))
    X, y, isr = [], [], []
    for smi, (ex, ce) in by.items():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        arr = np.zeros(1024, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m), arr)
        X.append(arr)
        if ex:
            y.append(float(np.median(ex))); isr.append(False)
        else:
            y.append(float(np.median(ce))); isr.append(True)
    if len(X) < 300:
        print(f"skip {tid}: n={len(X)}"); return None
    X = np.asarray(X, np.uint8); y = np.asarray(y); isr = np.asarray(isr, bool)
    if isr.sum() < 30 or (~isr).sum() < 150:
        print(f"skip {tid}: cens={isr.sum()} exact={(~isr).sum()}"); return None
    np.savez_compressed(f, X=X, y=y, isr=isr, name=rows[0][3])
    print(f"{tid} n={len(X)} cens={isr.sum()} ({time.time()-t0:.0f}s)", flush=True)
    return tid


if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        kept = [t for t in pool.map(one, [e[0] for e in eligible], chunksize=1) if t is not None]
    json.dump(kept, open(OUT / "kept.json", "w"))
    print(f"kept {len(kept)} in {time.time()-t0:.0f}s")
