"""
Summary counts of the red-team labels quoted in the article (Section 8.1).
Reads redteam.txt.gz (LANL; not redistributed) and writes
results/redteam_summary.json, which contains only aggregate counts.

Usage: python redteam_summary.py [redteam.txt.gz]
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "redteam.txt.gz")
rt = pd.read_csv(path, header=None, names=["t", "u", "s", "d"])
first = rt.groupby("d").t.min().sort_values() / 86400
top = rt.s.value_counts().index[0]
out = {"records": int(len(rt)), "source_hosts": int(rt.s.nunique()),
       "destination_hosts": int(rt.d.nunique()), "user_accounts": int(rt.u.nunique()),
       "top_source": top, "top_source_records": int((rt.s == top).sum()),
       "top_source_destinations": int(rt[rt.s == top].d.nunique()),
       "destinations_reached_by_day": {d: int((first <= d).sum()) for d in range(1, 31)}}
json.dump(out, open(os.path.join(HERE, "results", "redteam_summary.json"), "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "destinations_reached_by_day"}))
