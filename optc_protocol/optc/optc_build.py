"""
Stage 2 for OpTC: merge the per-file flow tables (optc_filter.py), remove
flows recorded at both ends, map addresses to hosts, and aggregate.

Deduplication and address mapping follow LMDEval's extract_optc.py: per
five-tuple (sip, dip, sport, dport, proto), if inbound and outbound records
balance, keep the outbound ones; if all records have one direction, keep
all; otherwise keep the inbound ones.  Addresses are mapped to host names
with optc_known_addresses.json plus the mappings learnt from the records
(outbound: sip -> reporting host; inbound: dip -> reporting host).  One
deviation, so that deduplication cannot drop a label: a five-tuple's label
is the largest label among its records, attached to the record kept.

Outputs (in --out):
  flows_dedup.parquet          ts, src, dst, dport, lab (lab: 2 lateral movement, 1 other red team, 0 none)
  edges_hour.csv.gz, edges_day.csv.gz   s, hour|day, d, count  (src != dst)
  hostunit_hour.csv.gz         per (s, hour): n, new_dst, k_dst, X_host, KL_host, H_today,
                               lm (labelled lateral-movement flows), rt (all red-team flows)
  hostunit_day.csv.gz          the same per (s, day)
  build_summary.json
Usage: python3 optc_build.py flows_dir labels_dir out_dir
"""
import glob
import json
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hostunit import host_unit_scores  # noqa: E402

FLOWS, LAB, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
files = sorted(f for f in glob.glob(os.path.join(FLOWS, "*.parquet")) if os.path.exists(f[:-8] + ".done"))
df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
for c in ("host", "sip", "dip"):
    df[c] = df[c].astype(str)
n_raw = len(df)

# address -> host map
addr = json.load(open(os.path.join(LAB, "optc_known_addresses.json")))
learnt = pd.concat([df.loc[df.outbound == 1, ["sip", "host"]].rename(columns={"sip": "ip"}),
                    df.loc[df.outbound == 0, ["dip", "host"]].rename(columns={"dip": "ip"})]).drop_duplicates()
n_conflict = int(learnt.groupby("ip")["host"].nunique().gt(1).sum())
for ip, h in learnt.itertuples(index=False):
    addr.setdefault(ip, h)

strip = lambda h: h.split(".")[0] if isinstance(h, str) else h  # noqa: E731

# two-ended deduplication
key = ["sip", "dip", "sport", "dport", "proto"]
df["sign"] = 2 * df["outbound"].astype("int8") - 1
g = df.groupby(key, sort=False, observed=True)
df["bal"] = g["sign"].transform("sum")
df["cnt"] = g["sign"].transform("size")
df["lab5"] = g["lab"].transform("max")
keep = ((df.bal == 0) & (df.outbound == 1)) | (df.bal.abs() == df.cnt) | \
       ((df.bal != 0) & (df.bal.abs() != df.cnt) & (df.outbound == 0))
# lab: five-tuple's largest label (primary, Amendment 1 A3); lab_kept: LMDEval's rule,
# the label of the kept record only (sensitivity)
d = df.loc[keep, ["ts", "sip", "dip", "dport", "lab5", "lab"]].rename(columns={"lab": "lab_kept", "lab5": "lab"})
reporting = sorted({strip(h) for h in df["host"].unique()})
d["src"] = d.sip.map(lambda a: strip(addr.get(a, a)))
d["dst"] = d.dip.map(lambda a: strip(addr.get(a, a)))
n_self = int((d.src == d.dst).sum())
d = d[d.src != d.dst]
d["hour"] = (d.ts // 3600).astype("int64")
d["day"] = (d.ts // 86400).astype("int64")
d[["ts", "src", "dst", "dport", "lab", "lab_kept"]].to_parquet(os.path.join(OUT, "flows_dedup.parquet"), index=False)

hosts = pd.Index(pd.unique(pd.concat([d.src, d.dst])))
K = len(hosts)
code = {h: i for i, h in enumerate(hosts)}
d["s"], d["d"] = d.src.map(code), d.dst.map(code)
summary = {"files": len(files), "records_internal_flow_start": n_raw, "after_dedup_and_self_loops": int(len(d)),
           "self_loops_dropped": n_self, "address_conflicts": n_conflict, "K_hosts": K,
           "labelled_lm_flows": int((d.lab == 2).sum()), "labelled_rt_flows": int((d.lab >= 1).sum()),
           "labelled_lm_flows_kept_rule": int((d.lab_kept == 2).sum()), "reporting_hosts": reporting,
           "days": sorted(int(x) for x in d.day.unique())}
pd.Series(hosts, name="host").to_csv(os.path.join(OUT, "host_codes.csv"), index_label="code")

for unit in ("hour", "day"):
    sd = d.groupby(["s", unit, "d"]).size().rename("count").reset_index()
    sd.to_csv(os.path.join(OUT, f"edges_{unit}.csv.gz"), index=False)
    hd = host_unit_scores(sd, unit, K)
    lab = d.assign(lm=(d.lab == 2).astype(int), rt=(d.lab >= 1).astype(int), lm_kept=(d.lab_kept == 2).astype(int)) \
        .groupby(["s", unit])[["lm", "rt", "lm_kept"]].sum()
    hd = hd.merge(lab.reset_index(), on=["s", unit], how="left").fillna({"lm": 0, "rt": 0, "lm_kept": 0})
    hd.to_csv(os.path.join(OUT, f"hostunit_{unit}.csv.gz"), index=False)
    summary[f"host_{unit}s"] = int(len(hd))
    summary[f"positive_{unit}s_lm"] = int((hd.lm > 0).sum())
    summary[f"positive_{unit}s_rt"] = int((hd.rt > 0).sum())
summary["lm_sources"] = sorted(hosts[int(s)] for s in d.loc[d.lab == 2, "s"].unique())
json.dump(summary, open(os.path.join(OUT, "build_summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))
