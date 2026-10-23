"""
Per-host information-theoretic detectors on LANL cyber1 (full campaign).

For each source computer s and day t, with destination counts c_t(s, d)
and the host's causal history (all days < t):
  * cross-entropy  X_t(s) = -(1/n_t(s)) * sum_d c_t(s,d) log2 p_<t(d|s),
    p_<t(d|s) = (c_<t(s,d) + a) / (c_<t(s) + a*K)   (Laplace, a = 1)
    -- mean surprisal of today's destinations under the host's own past;
  * total surprisal  S_t(s) = n_t(s) * X_t(s)  (volume of novel logons);
  * KL divergence    D_t(s) = X_t(s) - H_t(s) (today's entropy removed);
  * population surprisal  P_t(s): as S_t but with p_<t(d) the network-wide
    destination popularity, i.e. logons to rarely contacted hosts;
  * new destinations (first-ever pair), for comparison.
Positives as in cyber1_full.py
(metrics on campaign days; alert budget = top-b host-days per day).

Input : filt.csv, redteam.txt.gz.  Also writes srcday_dst_counts.csv.gz
        (s, day, d, count) so later variants need no re-parse.
Output: cyber1_perhost_results.json, hostday_perhost.csv.gz
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
FILT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "filt.csv")
RT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "redteam.txt.gz")
DAY = 86400
K = 17684
ALPHA = 1.0
LT_KEEP = {"Network", "RemoteInteractive"}
CAL_START, CAL_END = 31, 58
BUDGETS = (1, 10)
CACHE = os.path.join(HERE, "srcday_dst_counts.csv.gz")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def comp_id(s):
    return pd.to_numeric(s.str.slice(1), errors="coerce").fillna(-1).astype(np.int64)


if os.path.exists(CACHE):
    sd = pd.read_csv(CACHE)
    log("loaded cache", len(sd))
else:
    parts = []
    for ch in pd.read_csv(FILT, header=None, names=["time", "src", "dst", "lt"],
                          usecols=[0, 1, 2, 3], chunksize=20_000_000,
                          dtype={"time": np.int64, "src": str, "dst": str, "lt": str}):
        ch = ch[ch["lt"].isin(LT_KEEP)]
        g = pd.DataFrame({"s": comp_id(ch["src"]).to_numpy(), "day": (ch["time"] // DAY).to_numpy(),
                          "d": comp_id(ch["dst"]).to_numpy()}).value_counts().reset_index()
        parts.append(g)
        log("chunk", sum(len(p) for p in parts))
    sd = pd.concat(parts).groupby(["s", "day", "d"], sort=False)["count"].sum().reset_index()
    sd.to_csv(CACHE, index=False, compression="gzip")
    log("aggregated", len(sd))

sd = sd.sort_values(["s", "d", "day"]).reset_index(drop=True)
# causal per-(s,d) prior count: cumulative count over earlier days
sd["prior_sd"] = sd.groupby(["s", "d"])["count"].cumsum() - sd["count"]
# per-(s,day) totals and causal per-s prior totals
tot = sd.groupby(["s", "day"])["count"].sum().rename("n").reset_index().sort_values(["s", "day"])
tot["prior_s"] = tot.groupby("s")["n"].cumsum() - tot["n"]
sd = sd.merge(tot, on=["s", "day"])
# population popularity of destinations, causal by day
pop = sd.groupby(["d", "day"])["count"].sum().rename("dn").reset_index().sort_values(["d", "day"])
pop["prior_d"] = pop.groupby("d")["dn"].cumsum() - pop["dn"]
daytot = pop.groupby("day")["dn"].sum().sort_index()
prior_all = (daytot.cumsum() - daytot).rename("prior_all")
sd = sd.merge(pop[["d", "day", "prior_d"]], on=["d", "day"]).merge(prior_all, left_on="day", right_index=True)

lp_host = np.log2((sd["prior_sd"] + ALPHA) / (sd["prior_s"] + ALPHA * K))
lp_pop = np.log2((sd["prior_d"] + ALPHA) / (sd["prior_all"] + ALPHA * K))
p_today = sd["count"] / sd["n"]
sd["surp_host"] = -sd["count"] * lp_host
sd["surp_pop"] = -sd["count"] * lp_pop
sd["ent_today"] = -p_today * np.log2(p_today)
sd["new"] = (sd["prior_sd"] == 0).astype(int)
g = sd.groupby(["s", "day"])
hd = pd.DataFrame({"n": g["count"].sum(), "S_host": g["surp_host"].sum(), "S_pop": g["surp_pop"].sum(),
                   "H_today": g["ent_today"].sum(), "new_dst": g["new"].sum(),
                   "k_dst": g["d"].nunique()}).reset_index()
hd["X_host"] = hd["S_host"] / hd["n"]
hd["KL_host"] = hd["X_host"] - hd["H_today"]
log("host-days", len(hd))

rt = pd.read_csv(RT, header=None, names=["time", "user", "src", "dst"])
rt["s"], rt["day"] = comp_id(rt["src"]), rt["time"] // DAY
pos = rt.groupby(["s", "day"]).size().rename("rt").reset_index()
hd = hd.merge(pos, on=["s", "day"], how="left").fillna({"rt": 0})
hd["label"] = (hd["rt"] > 0).astype(int)
hd.to_csv(os.path.join(HERE, "hostday_perhost.csv.gz"), index=False, compression="gzip")

first_day, last_day = int(rt.time.min() // DAY), int(rt.time.max() // DAY)
camp = ((hd.day >= first_day) & (hd.day <= last_day)).to_numpy()
cal = ((hd.day >= CAL_START) & (hd.day < CAL_END)).to_numpy()
n_cal_days = hd.loc[cal, "day"].nunique()
lab = hd["label"].to_numpy()
res = {"n_host_days_campaign": int(camp.sum()), "positives": int(lab[camp].sum())}
for sc in ("X_host", "S_host", "KL_host", "S_pop", "H_today", "new_dst", "k_dst", "n"):
    s = hd[sc].to_numpy(float)
    r = {"auc": float(roc_auc_score(lab[camp], s[camp])),
         "ap": float(average_precision_score(lab[camp], s[camp]))}
    # Alert budget enforced per day: the analyst sees the day's top-b
    # host-days by score (ties broken by order), so every score spends
    # exactly the same number of alerts.
    rank = hd.assign(_s=s).groupby("day")["_s"].rank(ascending=False, method="first").to_numpy()
    for b in BUDGETS:
        fl = rank <= b
        r[f"recall_at_{b}"] = float(fl[camp & (lab == 1)].mean())
        r[f"precision_at_{b}"] = float(lab[camp & fl].mean())
        det = hd.loc[fl & (lab == 1)]
        r[f"sources_detected_at_{b}"] = sorted(int(x) for x in det.s.unique())
        r[f"first_detection_day_at_{b}"] = float(det.day.min()) if len(det) else None
    res[sc] = r
    log(sc, {k: v for k, v in r.items() if not isinstance(v, list)})
with open(os.path.join(HERE, "cyber1_perhost_results.json"), "w") as f:
    json.dump(res, f, indent=2)
log("saved")
