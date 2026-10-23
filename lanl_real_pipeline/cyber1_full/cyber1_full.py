"""
Full-campaign evaluation on LANL cyber1 (Kent 2015): do global entropy
detectors, or per-host statistics, respond to the red-team campaign?

Input  : filt.csv  (time,src,dst,logon_type) from filter_auth.sh
         redteam.txt.gz (time,user@domain,src,dst; 749 events)
Output : cyber1_full_results.json, hourly.csv, hostday_scores.csv.gz

Event set: successful LogOn events between distinct computers with logon
type Network or RemoteInteractive (as in the paper); red-team coverage is
reported for this set and for all logon types.

Global (hourly) observables: destination-host Shannon entropy H_dst,
number of distinct destinations, number of new (src,dst) edges.
Detectors (as in the paper): level, CUSUM (k=0.5), rate (4-hour... see
RATE_WIN) on the standardised series, reset + refractory after each alarm.
Calibration period: days CAL_START..CAL_END (after the last red-team event),
thresholds set to TARGET_FAR alarms/day there.  Two standardisations:
global mean/sd, and hour-of-day mean/sd (diurnal-adjusted).

Per-host (source computer, day) scores: distinct destinations, new
destinations (first-ever (src,dst) pair), destination entropy.  Positives:
host-days with >= 1 red-team event as source.  Metrics: ROC AUC, average
precision, and recall at a budget of ALERTS_PER_DAY alerts/day, the
threshold being set on the calibration days.

The article does not report the global detectors of this script: their
calibration on the post-campaign days 31-58 is biased by the lower entropy
of the campaign period.  It reports cyber1_hourly_checks.py, which
calibrates on red-team-free hours of the whole period, and the per-host
scores of cyber1_perhost.py.  This script provides hourly.csv.
"""
import gzip
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
OUT = HERE
DAY, HOUR = 86400, 3600
LT_KEEP = {"Network", "RemoteInteractive"}
CAL_START, CAL_END = 31.0, 58.0          # days, after last red-team event (29.6)
TARGET_FAR = 0.1                          # 27 calibration days: 0.01/day is not estimable
ALERTS_PER_DAY = (1, 10)
RATE_WIN = 4                              # hours, rate window and refractory
K_CUSUM = 0.5
CHUNK = 20_000_000


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def comp_id(s):
    """'C1234' -> 1234; anything else -> -1."""
    return pd.to_numeric(s.str.slice(1), errors="coerce").fillna(-1).astype(np.int64)


# --------------------------------------------------------------------------
# Red-team labels
rt = pd.read_csv(RT, header=None, names=["time", "user", "src", "dst"])
rt["src_i"], rt["dst_i"] = comp_id(rt["src"]), comp_id(rt["dst"])
rt_keys = set(zip(rt["time"], rt["src_i"], rt["dst_i"]))
log("red-team events", len(rt), "first day %.3f last day %.3f" % (rt.time.min() / DAY, rt.time.max() / DAY))

# --------------------------------------------------------------------------
# Single streaming pass: hourly aggregates, edge first-seen, host-day counts
hour_dst = []        # (hour, dst, count)
hour_n = []          # (hour, n)
edge_first = []      # (src, dst, first time) per chunk
hostday_dst = []     # (src, day, dst, count)
rt_found = {"keep_set": 0, "all_types": 0}
lt_counts = {}
n_total = n_keep = 0
for ch in pd.read_csv(FILT, header=None, names=["time", "src", "dst", "lt"],
                      chunksize=CHUNK, dtype={"time": np.int64, "src": str, "dst": str, "lt": str}):
    ch["src_i"], ch["dst_i"] = comp_id(ch["src"]), comp_id(ch["dst"])
    n_total += len(ch)
    for k, v in ch["lt"].value_counts().items():
        lt_counts[k] = lt_counts.get(k, 0) + int(v)
    keys = pd.MultiIndex.from_arrays([ch["time"], ch["src_i"], ch["dst_i"]])
    is_rt = keys.isin(list(rt_keys)) if rt_keys else np.zeros(len(ch), bool)
    rt_found["all_types"] += int(is_rt.sum())
    keep = ch["lt"].isin(LT_KEEP).to_numpy()
    rt_found["keep_set"] += int((is_rt & keep).sum())
    c = ch.loc[keep, ["time", "src_i", "dst_i"]]
    n_keep += len(c)
    hr = (c["time"] // HOUR).to_numpy()
    dy = (c["time"] // DAY).to_numpy()
    hd = pd.DataFrame({"h": hr, "d": c["dst_i"].to_numpy()}).value_counts().reset_index()
    hour_dst.append(hd)
    hour_n.append(pd.Series(hr).value_counts())
    ef = c.groupby(["src_i", "dst_i"], sort=False)["time"].min().reset_index()
    edge_first.append(ef)
    sd = pd.DataFrame({"s": c["src_i"].to_numpy(), "day": dy, "d": c["dst_i"].to_numpy()}).value_counts().reset_index()
    hostday_dst.append(sd)
    log(f"rows {n_total:,} kept {n_keep:,} rt_found {rt_found}")

log("combining")
hd = pd.concat(hour_dst).groupby(["h", "d"], sort=False)["count"].sum().reset_index()
hn = pd.concat(hour_n).groupby(level=0).sum().sort_index()
ef = pd.concat(edge_first).groupby(["src_i", "dst_i"], sort=False)["time"].min().reset_index()
sd = pd.concat(hostday_dst).groupby(["s", "day", "d"], sort=False)["count"].sum().reset_index()
del hour_dst, edge_first, hostday_dst


def entropy_by(df, key, cnt="count"):
    tot = df.groupby(key)[cnt].transform("sum")
    p = df[cnt] / tot
    return (-(p * np.log2(p))).groupby(df[key] if isinstance(key, str) else [df[k] for k in key]).sum()


# --------------------------------------------------------------------------
# Hourly global series
hours = np.arange(int(hn.index.max()) + 1)
H = entropy_by(hd, "h").reindex(hours)
K = hd.groupby("h")["d"].nunique().reindex(hours)
ef["h"] = ef["time"] // HOUR
NE = ef.groupby("h").size().reindex(hours).fillna(0)
hourly = pd.DataFrame({"hour": hours, "day": hours / 24.0, "n": hn.reindex(hours).fillna(0).to_numpy(),
                       "H_dst": H.to_numpy(), "k_dst": K.to_numpy(), "new_edges": NE.to_numpy()})
rt_h = (rt["time"] // HOUR).value_counts()
hourly["rt_events"] = rt_h.reindex(hours).fillna(0).to_numpy()
hourly.to_csv(os.path.join(OUT, "hourly.csv"), index=False)
log("hourly series", len(hourly))


def alarms(x, kind, h, start=0):
    out = []
    if kind == "CUSUM":
        pos = start
        while pos < len(x):
            S = np.cumsum(np.nan_to_num(x[pos:], nan=0.0) - K_CUSUM)
            g = S - np.minimum(0.0, np.minimum.accumulate(S))
            hit = np.flatnonzero(g > h)
            if hit.size == 0:
                break
            a = pos + int(hit[0]); out.append(a); pos = a + RATE_WIN
        return out
    exc = np.flatnonzero(np.nan_to_num(x[start:], nan=-np.inf) > h) + start
    nxt = start
    for e in exc:
        if e >= nxt:
            out.append(int(e)); nxt = e + RATE_WIN
    return out


def standardise(v, day, hod, cal, diurnal):
    z = np.full(len(v), np.nan)
    if diurnal:
        for k in range(24):
            m = (hod == k)
            mu, sd = np.nanmean(v[m & cal]), np.nanstd(v[m & cal])
            z[m] = (v[m] - mu) / (sd if sd > 0 else 1.0)
    else:
        z = (v - np.nanmean(v[cal])) / np.nanstd(v[cal])
    return z


def rate_stat(z):
    w = np.arange(RATE_WIN) - (RATE_WIN - 1) / 2.0
    w = w / np.sum(w ** 2)
    s = np.full(len(z), np.nan)
    s[RATE_WIN - 1:] = np.convolve(np.nan_to_num(z), w[::-1], mode="valid")
    return s


day = hourly["day"].to_numpy(); hod = (hourly["hour"] % 24).to_numpy()
cal = (day >= CAL_START) & (day < CAL_END)
camp = (day >= rt.time.min() / DAY) & (day <= rt.time.max() / DAY)
pre = day < rt.time.min() / DAY
onset_h = int(rt.time.min() // HOUR)
glob = {}
for obs in ("H_dst", "k_dst", "new_edges", "n"):
    v = hourly[obs].to_numpy(float)
    for diurnal in (False, True):
        z = standardise(v, day, hod, cal, diurnal)
        stats = {"LEVEL": z, "CUSUM": z, "RATE": rate_stat(z) / np.nanstd(rate_stat(z)[cal])}
        for kind, x in stats.items():
            xc = x.copy(); xc[~cal] = np.nan
            cal_days = cal.sum() / 24.0
            lo, hi = 0.0, 100.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                n_al = len([a for a in alarms(np.where(cal, x, np.nan), kind, mid) if cal[a]])
                lo, hi = (mid, hi) if n_al / cal_days > TARGET_FAR else (lo, mid)
            al = np.array(alarms(x, kind, hi))
            in_camp = al[camp[al]] if al.size else al
            first_after = al[al >= onset_h] if al.size else al
            glob[f"{obs}|{'diurnal' if diurnal else 'global'}|{kind}"] = {
                "threshold": hi,
                "alarms_per_day_calibration": float(np.sum(cal[al]) / cal_days) if al.size else 0.0,
                "alarms_per_day_campaign": float(len(in_camp) / (camp.sum() / 24.0)),
                "alarms_per_day_precampaign": float(np.sum(pre[al]) / max(pre.sum() / 24.0, 1e-9)) if al.size else 0.0,
                "first_alarm_after_onset_hours": float(first_after[0] - onset_h) if first_after.size else None,
                "alarms_in_hours_with_redteam": int(np.sum(hourly["rt_events"].to_numpy()[al] > 0)) if al.size else 0,
            }
log("global detectors done")

# --------------------------------------------------------------------------
# Per-host-day scores
g = sd.groupby(["s", "day"])
hostday = pd.DataFrame({"k_dst": g["d"].nunique(), "n": g["count"].sum()})
p = sd["count"] / g["count"].transform("sum")
hostday["H_dst"] = (-(p * np.log2(p))).groupby([sd["s"], sd["day"]]).sum()
ef["day"] = ef["time"] // DAY
hostday["new_dst"] = ef.groupby(["src_i", "day"]).size().rename_axis(["s", "day"]).reindex(hostday.index).fillna(0)
hostday = hostday.reset_index()
rt["day"] = rt["time"] // DAY
pos = rt.groupby(["src_i", "day"]).size().rename("rt").reset_index().rename(columns={"src_i": "s"})
hostday = hostday.merge(pos, on=["s", "day"], how="left").fillna({"rt": 0})
hostday["label"] = (hostday["rt"] > 0).astype(int)
hostday.to_csv(os.path.join(OUT, "hostday_scores.csv.gz"), index=False, compression="gzip")
log("host-days", len(hostday), "positives", int(hostday.label.sum()))

per_host = {}
camp_days = (hostday["day"] >= int(rt.time.min() // DAY)) & (hostday["day"] <= int(rt.time.max() // DAY))
cal_hd = (hostday["day"] >= CAL_START) & (hostday["day"] < CAL_END)
n_cal_days = hostday.loc[cal_hd, "day"].nunique()
for sc in ("k_dst", "new_dst", "H_dst", "n"):
    s = hostday[sc].to_numpy(float)
    m = camp_days.to_numpy()
    res = {"auc_campaign_days": float(roc_auc_score(hostday.label[m], s[m])),
           "ap_campaign_days": float(average_precision_score(hostday.label[m], s[m])),
           "positives": int(hostday.label[m].sum()), "host_days": int(m.sum())}
    for b in ALERTS_PER_DAY:
        thr = np.sort(s[cal_hd.to_numpy()])[::-1][min(b * n_cal_days, cal_hd.sum() - 1)]
        flagged = s > thr
        res[f"recall_at_{b}_alerts_per_day"] = float(flagged[m & (hostday.label == 1).to_numpy()].mean())
        res[f"alerts_per_day_campaign_at_{b}"] = float(flagged[m].sum() / hostday.loc[m, "day"].nunique())
        first = hostday.loc[flagged & (hostday.label == 1).to_numpy(), "day"]
        res[f"first_detected_day_at_{b}"] = float(first.min()) if len(first) else None
    per_host[sc] = res
log("per-host done")

out = {"n_events_success_logon_distinct": n_total, "n_events_network_remoteinteractive": n_keep,
       "logon_type_counts": lt_counts,
       "redteam": {"n": len(rt), "found_all_types": rt_found["all_types"], "found_keep_set": rt_found["keep_set"],
                   "first_day": float(rt.time.min() / DAY), "last_day": float(rt.time.max() / DAY),
                   "n_src_hosts": int(rt.src_i.nunique()), "n_dst_hosts": int(rt.dst_i.nunique())},
       "calibration_days": [CAL_START, CAL_END], "target_far_per_day": TARGET_FAR,
       "global_detectors": glob, "per_host": per_host}
with open(os.path.join(OUT, "cyber1_full_results.json"), "w") as f:
    json.dump(out, f, indent=2)
log("saved")
