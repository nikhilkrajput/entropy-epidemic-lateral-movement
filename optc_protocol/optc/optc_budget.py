"""
Held-out OpTC evaluation, exactly as fixed in practical/PREREGISTRATION.md
(the script refuses to run if the protocol file's SHA-256 differs from the
hash recorded before the data were read).

Input : <build_dir>/hostunit_hour.csv.gz, hostunit_day.csv.gz, edges_hour.csv.gz
        and powell_hour.csv.gz, powell_day.csv.gz (powell_baseline.py on the edges)
        from optc_build.py; <build_dir>/build_summary.json.
Output: practical/results/optc_budget.json (or --out), and
        <build_dir>/eval_hour_primary.csv.gz for the figure.

Primary analysis : unit = source host x hour, labels = LMDEval lateral
                   movement, budgets b = 1, 10, 100 host-hours per day.
Secondary        : unit = host x day (b = 1, 10); labels = all red-team flows.
Evaluation days  : 23, 24, 25 Sep 2019 (local time); history from all earlier units.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PR = os.path.dirname(HERE)
sys.path.insert(0, PR)
import budget_eval as be  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("build_dir")
ap.add_argument("--prereg", default=os.path.join(PR, "PREREGISTRATION.md"))
ap.add_argument("--hash", default=os.path.join(PR, "results", "preregistration_sha256.txt"))
ap.add_argument("--out", default=os.path.join(PR, "results", "optc_budget.json"))
args = ap.parse_args()

frozen = dict((ln.split()[1], ln.split()[0]) for ln in open(args.hash) if ln.strip())
hashes = {}
for f in ("PREREGISTRATION.md", "PREREGISTRATION_AMENDMENT_1.md"):
    got = hashlib.sha256(open(os.path.join(os.path.dirname(args.prereg), f), "rb").read()).hexdigest()
    assert got == frozen[f], f"{f} changed since it was frozen ({got} != {frozen[f]})"
    hashes[f] = got

A = 1e5
D0 = (dt.date(2019, 9, 23) - dt.date(1970, 1, 1)).days
EVAL = (D0, D0 + 2)
SCORES = ["E", "E_ainf", "X_host", "KL_host", "new_dst", "H_today", "k_dst", "powell", "host_age", "neg_log_hist",
          "new_host_k"]
summary = json.load(open(os.path.join(args.build_dir, "build_summary.json")))
codes = pd.read_csv(os.path.join(args.build_dir, "host_codes.csv"))
name = dict(zip(codes["code"], codes["host"]))


def prepare(unit):
    hd = pd.read_csv(os.path.join(args.build_dir, f"hostunit_{unit}.csv.gz"))
    pw = pd.read_csv(os.path.join(args.build_dir, f"powell_{unit}.csv.gz"))
    hd = hd.merge(pw, on=["s", unit], how="left")
    # history and population rate run over units (hours or days) ...
    hd = hd.rename(columns={unit: "day"})
    hd = be.novelty_baselines(be.add_history(hd))
    hd["E"] = be.eb_novelty(hd, A)
    hd["E_ainf"] = be.eb_novelty(hd, np.inf)
    # ... budgets over calendar days
    hd = hd.rename(columns={"day": "unit"})
    hd["day"] = hd["unit"] // 24 if unit == "hour" else hd["unit"]
    return hd


def run(hd, label_col, budgets):
    c = hd[hd["day"].between(*EVAL)].copy()
    c["label"] = (c[label_col] > 0).astype(int)
    c["events"] = c[label_col]
    r = be.evaluate(c, SCORES, budgets=budgets)
    r["within_day_auc"] = be.within_day_auc(c, SCORES)
    r["source_names"] = {int(s): name[int(s)] for s in r["sources"]}
    for sc in SCORES:
        for b in budgets:
            r[sc][f"b{b}"]["sources_detected_names"] = [name[s] for s in r[sc][f"b{b}"]["sources_detected"]]
    return r, c


out = {"protocol_sha256": hashes, "a": A, "eval_days": list(EVAL), "build": summary}
hour = prepare("hour")
out["primary_hour_lm"], c = run(hour, "lm", (1, 10, 100))
c.to_csv(os.path.join(args.build_dir, "eval_hour_primary.csv.gz"), index=False)
out["secondary_hour_rt"], _ = run(hour, "rt", (1, 10, 100))
out["sensitivity_hour_lm_kept_rule"], _ = run(hour, "lm_kept", (1, 10, 100))
day = prepare("day")
out["secondary_day_lm"], _ = run(day, "lm", (1, 10))
out["secondary_day_rt"], _ = run(day, "rt", (1, 10))
# threshold rule over hourly units: tau from the b-th largest host-hour score of earlier days
hh = hour.assign(label=(hour["lm"] > 0).astype(int))
out["threshold_rule_hour_lm"] = {f"b{b}": be.threshold_rule(hh, "E", b, eval_days=EVAL) for b in (10, 100)}

p = out["primary_hour_lm"]
# Amendment 1, A4: sources with logs = labelled sources that report in the filtered records
with_logs = [s for s in p["sources"] if name[s] in set(summary["reporting_hosts"])]
det10 = set(p["E"]["b10"]["sources_detected"])
out["hypotheses"] = {
    "H1_E_recall10_gt_X": p["E"]["b10"]["recall"] > p["X_host"]["b10"]["recall"],
    "H2_E_recall10_gt_new_dst": p["E"]["b10"]["recall"] > p["new_dst"]["b10"]["recall"],
    "H3_E_detects_half_sources_with_logs_b10": len(det10 & set(with_logs)) >= len(with_logs) / 2,
    "sources_with_logs": [name[s] for s in with_logs],
}
json.dump(out, open(args.out, "w"), indent=2)
for sc in SCORES:
    print(f"{sc:12s} auc {p['within_day_auc'][sc]:.3f} " + " ".join(
        f"b{b}: R={p[sc][f'b{b}']['recall']:.3f} src={p[sc][f'b{b}']['n_sources_detected']}" for b in (1, 10, 100)))
print(out["hypotheses"])
