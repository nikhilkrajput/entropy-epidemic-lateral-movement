"""
Alert-budget curves for the per-host scores (results/hostday_perhost.csv.gz).

For each score: within-day percentile rank of every red-team host-day
(campaign days), the within-day-percentile ROC AUC (robust to day-to-day
level changes), and the daily alert budget b needed so that the day's top-b
host-days contain 25%, 50% and 75% of red-team host-days.
Also scores two host-novelty baselines (see cyber1_novelty.py): host_age,
minus the days since the host was first seen as a source, and
neg_log_hist, -log(1 + logons in the host's earlier history).
Writes results/cyber1_budget_curve.json.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
hd = pd.read_csv(os.path.join(RES, "hostday_perhost.csv.gz")).sort_values(["s", "day"])
hd["host_age"] = -(hd["day"] - hd.groupby("s")["day"].transform("min")).astype(float)
hd["neg_log_hist"] = -np.log1p(hd.groupby("s")["n"].cumsum() - hd["n"])
first, last = int(hd.loc[hd.label == 1, "day"].min()), int(hd.loc[hd.label == 1, "day"].max())
c = hd[(hd.day >= first) & (hd.day <= last)].copy()
out = {"campaign_days": [first, last], "host_days": int(len(c)), "positives": int(c.label.sum()),
       "median_host_days_per_day": float(c.groupby("day").size().median())}
for sc in ("X_host", "KL_host", "H_today", "new_dst", "k_dst", "S_host", "S_pop", "n",
           "host_age", "neg_log_hist"):
    c["rank"] = c.groupby("day")[sc].rank(ascending=False, method="average")
    c["pct"] = c.groupby("day")[sc].rank(pct=True)
    pos = c[c.label == 1]
    r = {"auc_pooled": float(roc_auc_score(c.label, c[sc])),
         "auc_within_day_percentile": float(roc_auc_score(c.label, c["pct"])),
         "median_rank_of_positives": float(pos["rank"].median()),
         "best_rank_of_positives": float(pos["rank"].min()),
         "recall_at_10": float(np.mean(pos["rank"] <= 10))}
    for q in (0.25, 0.5, 0.75):
        r[f"budget_for_{int(q * 100)}pct_recall"] = float(np.quantile(pos["rank"], q))
    out[sc] = r
    print(sc, {k: round(v, 3) for k, v in r.items()})
json.dump(out, open(os.path.join(RES, "cyber1_budget_curve.json"), "w"), indent=2)
