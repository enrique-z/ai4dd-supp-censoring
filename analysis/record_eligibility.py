#!/usr/bin/env python3
"""Record how many targets meet the pre-registered S4 inclusion rule.

The number belongs in the paper (it is the denominator that makes "we analysed
every eligible target" checkable) but the confirmatory S4 run now in flight
predates `n_eligible` being written into its provenance. Rather than hand-edit
an evidence file -- which would defeat the point of having one -- this script
re-derives the count from the database with the identical query and writes it as
its own small evidence artifact.

Usage: python3 analysis/record_eligibility.py
"""
from __future__ import annotations
import json, sqlite3, time
from pathlib import Path

DB = Path("./data/chembl_36/"
          "chembl_36_sqlite/chembl_36.db")
OUT = Path(__file__).resolve().parent.parent / "evidence" / "eligibility_raw.json"

Q = """select a.tid, count(*) c,
              sum(case when act.standard_relation in ('>','>=','>>') then 1 else 0 end) r
       from activities act
       join assays a on act.assay_id = a.assay_id
       join target_dictionary td on a.tid = td.tid
       where act.standard_type='IC50' and act.standard_units='nM'
         and act.standard_value > 0 and td.target_type='SINGLE PROTEIN'
       group by a.tid having c > 2000 order by c desc"""


def main() -> int:
    cur = sqlite3.connect(f"file:{DB}?mode=ro", uri=True).cursor()
    cur.execute("PRAGMA temp_store=MEMORY")
    rows = list(cur.execute(Q))
    eligible = [x for x in rows if x[2] >= 200]
    OUT.write_text(json.dumps({
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query": " ".join(Q.split()),
        "inclusion_rule": "single protein, >2000 IC50 nM records, >=200 right-censored",
        "n_over_2000_records": len(rows),
        "n_eligible": len(eligible),
    }, indent=1))
    print(f"{len(rows)} targets with >2000 records; {len(eligible)} eligible")
    print(f"wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
