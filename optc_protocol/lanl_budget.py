"""
LANL 2015 (cyber1): fixed-budget comparison of per-host scores.

Input : lanl_real_pipeline/cyber1_full/results/hostday_perhost.csv.gz
        (written by cyber1_perhost.py from all 366.6 M network and
        remote-interactive logons; 'rt' = red-team records of the source
        that day, 'label' = rt > 0).
Output: practical/results/lanl_budget.json

LANL is the development set: the prior weight a of the empirical-Bayes
novelty score is chosen here (largest mean recall over b = 1, 10, 100;
ties to the smaller a) and then frozen for OpTC (PREREGISTRATION.md).
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import budget_eval as be  # noqa: E402

ROOT = os.path.join(HERE, "..")
SRC = os.path.join(ROOT, "lanl_real_pipeline", "cyber1_full", "results", "hostday_perhost.csv.gz")
OUT = os.path.join(HERE, "results")
A_GRID = (1, 10, 100, 1000, 10000, 100000, float("inf"))   # inf: network rate only, no host history

POWELL = os.path.join(OUT, "powell_lanl.csv.gz")   # powell_baseline.py on srcday_dst_counts.csv.gz (server)

hd = pd.read_csv(SRC).rename(columns={"rt": "events"})
hd = be.novelty_baselines(be.add_history(hd))
pw = pd.read_csv(POWELL)
hd = hd.merge(pw, on=["s", "day"], how="left")
assert hd["powell"].notna().sum() == (hd.day > hd.day.min()).sum(), "Powell scores missing for some host-days"
def tag(a):
    return "inf" if a == float("inf") else str(int(a))


for a in A_GRID:
    hd[f"eb_a{tag(a)}"] = be.eb_novelty(hd, a)
first, last = int(hd.loc[hd.label == 1, "day"].min()), int(hd.loc[hd.label == 1, "day"].max())
c = hd[(hd.day >= first) & (hd.day <= last)].copy()

base = ["X_host", "KL_host", "new_dst", "k_dst", "H_today", "host_age", "neg_log_hist", "new_host_k", "powell"]
eb_cols = [f"eb_a{tag(a)}" for a in A_GRID]
res = be.evaluate(c, base + eb_cols)
res["within_day_auc"] = be.within_day_auc(c, base + eb_cols)
res["first_day_in_log"] = int(hd.day.min())
res["eval_days"] = [first, last]
sel = {tag(a): sum(res[f"eb_a{tag(a)}"][f"b{b}"]["recall"] for b in be.BUDGETS) / len(be.BUDGETS) for a in A_GRID}
best = max(A_GRID, key=lambda a: (round(sel[tag(a)], 12), -a))
res["eb_selection"] = {"criterion": "mean recall over b=1,10,100", "mean_recall": sel, "chosen_a": tag(best)}
chosen = f"eb_a{tag(best)}"
res["threshold_rule"] = {f"b{b}": be.threshold_rule(hd, chosen, b, eval_days=(first, last)) for b in (10, 100)}
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(os.path.join(OUT, "lanl_budget.json"), "w"), indent=2)
for sc in base + eb_cols:
    print(f"{sc:13s} auc {res['within_day_auc'][sc]:.3f} " + " ".join(
        f"b{b}: R={res[sc][f'b{b}']['recall']:.3f} [{res[sc][f'b{b}']['recall_ci95'][0]:.2f},"
        f"{res[sc][f'b{b}']['recall_ci95'][1]:.2f}] src={res[sc][f'b{b}']['n_sources_detected']} "
        f"cov={res[sc][f'b{b}']['event_cover_after_alert_day']:.2f}" for b in be.BUDGETS))
print("chosen a:", best, sel)
print("threshold rule:", res["threshold_rule"])
