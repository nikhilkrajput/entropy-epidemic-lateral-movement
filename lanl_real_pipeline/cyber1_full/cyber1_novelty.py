"""
Host-novelty controls for the per-host scores (results/hostday_perhost.csv.gz).

The red-team source hosts first appear as sources on the first campaign day,
so a per-host score may separate them simply because they are new.  This
script compares the cross-entropy and KL scores with two novelty baselines
and repeats the comparison on host-days with established history.

  host_age      days since the host was first seen as a source (low = new;
                scored as -age)
  neg_log_hist  -log(1 + logons in the host's history before today)

Analyses (campaign days, within-day percentile AUC as in
cyber1_budget_curve.py):
  * all host-days;
  * host-days with at least H_MIN_DAYS of history (excludes each host's
    first days);
  * partial ranking: AUC of the score among host-days in the same host-age
    stratum (age 0, 1-2, 3-6, 7+ days);
  * uncertainty: the 29 positives come from 4 source hosts, so the 95%
    interval resamples source hosts (cluster bootstrap, 2000 draws), and a
    leave-one-source-out AUC is reported.
Writes results/cyber1_novelty.json.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
H_MIN_DAYS = 3
N_BOOT = 2000

hd = pd.read_csv(os.path.join(RES, "hostday_perhost.csv.gz")).sort_values(["s", "day"])
hd["first_day"] = hd.groupby("s")["day"].transform("min")
hd["age"] = hd["day"] - hd["first_day"]
hd["hist_n"] = hd.groupby("s")["n"].cumsum() - hd["n"]
hd["host_age"] = -hd["age"].astype(float)
hd["neg_log_hist"] = -np.log1p(hd["hist_n"])
first, last = int(hd.loc[hd.label == 1, "day"].min()), int(hd.loc[hd.label == 1, "day"].max())
c = hd[(hd.day >= first) & (hd.day <= last)].copy()
SCORES = ("X_host", "KL_host", "host_age", "neg_log_hist", "new_dst", "H_today")
for sc in SCORES:
    c["pct_" + sc] = c.groupby("day")[sc].rank(pct=True)
pos = c[c.label == 1]
src = sorted(pos.s.unique())


def auc(d, sc):
    return float(roc_auc_score(d.label, d["pct_" + sc])) if d.label.nunique() == 2 else None


def cluster_ci(d, sc, rng):
    neg = d[d.label == 0]
    by_src = {s: d[(d.label == 1) & (d.s == s)] for s in src}
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(src, len(src), replace=True)
        p = pd.concat([by_src[s] for s in pick])
        if len(p):
            vals.append(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(p))],
                                      np.r_[neg["pct_" + sc].values, p["pct_" + sc].values]))
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


out = {"campaign_days": [first, last], "positives": int(len(pos)), "source_hosts": [int(s) for s in src],
       "positives_per_source": {int(s): int((pos.s == s).sum()) for s in src},
       "positive_age_days": sorted(int(a) for a in pos.age),
       "source_first_seen_day": {int(s): int(hd.loc[hd.s == s, "first_day"].iloc[0]) for s in src},
       "h_min_days": H_MIN_DAYS, "n_boot": N_BOOT}
est = c[c.age >= H_MIN_DAYS]
out["positives_with_history"] = int(est.label.sum())
strata = pd.cut(c.age, [-1, 0, 2, 6, np.inf], labels=["0", "1-2", "3-6", "7+"])
rng = np.random.default_rng(0)
for sc in SCORES:
    r = {"auc_all": auc(c, sc), "auc_all_source_ci95": cluster_ci(c, sc, rng),
         "auc_with_history": auc(est, sc),
         "auc_leave_one_source_out": {int(s): auc(c[(c.label == 0) | (c.s != s)], sc) for s in src}}
    # within host-age stratum: pool pairs (positive, negative) from the same stratum
    num = den = 0.0
    for lab in strata.cat.categories:
        d = c[strata == lab]
        npos, nneg = int(d.label.sum()), int((d.label == 0).sum())
        if npos and nneg:
            num += roc_auc_score(d.label, d["pct_" + sc]) * npos * nneg
            den += npos * nneg
    r["auc_within_age_stratum"] = float(num / den)
    out[sc] = r
    print(sc, {k: v for k, v in r.items() if k != "auc_leave_one_source_out"}, flush=True)
json.dump(out, open(os.path.join(RES, "cyber1_novelty.json"), "w"), indent=2)
