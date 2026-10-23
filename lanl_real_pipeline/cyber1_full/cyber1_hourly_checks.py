"""
Robustness checks on the hourly LANL red-team series (results/hourly.csv).

1. Level-shift-free calibration.  Entropy is lower during the campaign
   period than after it, so thresholds set on days 31-58 under-alarm in the
   campaign.  Here each hour is standardised against all red-team-free
   hours in the same hour-of-day x day-of-week cell (whole 58 days), the
   level/CUSUM/rate detectors are run on that series, thresholds are set
   so that red-team-free hours produce 0.1 alarms/day, and we count alarms
   in red-team hours against the rate in red-team-free campaign hours.
2. Mann-Whitney test of the same residuals: red-team hours vs other
   campaign hours.
3. Timing: ROC AUC of red-team hours vs other campaign hours for the raw
   hourly value, for the expected value of its hour-of-day x day-of-week
   cell alone (red-team-free hours; this carries only when the hour is),
   and for the residual.
4. Autocorrelation of hourly destination entropy after removing the mean
   of its hour-of-day x day-of-week cell (lags 1, 6 and 24 hours), for
   comparison with the simulator's AR(1) baseline.
Writes results/hourly_checks.json.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, binomtest
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ONSET, END = 1.7463541666666667, 29.59545138888889
RW, K_CUSUM, TARGET = 4, 0.5, 0.1

h = pd.read_csv(os.path.join(RES, "hourly.csv"))
h["hod"], h["dow"] = h.hour % 24, (h.hour // 24) % 7
rt = (h.rt_events > 0).to_numpy()
camp = ((h.day >= ONSET) & (h.day <= END)).to_numpy()


def cell_mean(col):
    ref = h[~rt].groupby(["hod", "dow"])[col].mean()
    return h.set_index(["hod", "dow"]).index.map(ref).to_numpy()


def resid(col):
    ref = h[~rt].groupby(["hod", "dow"])[col].agg(["mean", "std"])
    idx = h.set_index(["hod", "dow"]).index
    return ((h[col].to_numpy() - idx.map(ref["mean"]).to_numpy()) / idx.map(ref["std"]).to_numpy())


def auc_rt(x):
    sel = camp & ~np.isnan(x)
    return float(roc_auc_score(rt[sel], x[sel]))


def alarms(x, kind, thr):
    out = []
    if kind == "CUSUM":
        pos = 0
        while pos < len(x):
            S = np.cumsum(np.nan_to_num(x[pos:]) - K_CUSUM)
            g = S - np.minimum(0.0, np.minimum.accumulate(S))
            hit = np.flatnonzero(g > thr)
            if not hit.size:
                break
            a = pos + int(hit[0]); out.append(a); pos = a + RW
        return np.array(out, int)
    exc = np.flatnonzero(np.nan_to_num(x, nan=-np.inf) > thr)
    nxt = 0
    for e in exc:
        if e >= nxt:
            out.append(int(e)); nxt = e + RW
    return np.array(out, int)


def rate_stat(z):
    w = np.arange(RW) - (RW - 1) / 2.0
    w = w / np.sum(w ** 2)
    s = np.full(len(z), np.nan)
    s[RW - 1:] = np.convolve(np.nan_to_num(z), w[::-1], mode="valid")
    return s / np.nanstd(s[~rt])


free_days = (~rt).sum() / 24.0
out = {"n_hours": int(len(h)), "redteam_hours": int(rt.sum()),
       "redteam_share_of_campaign_hours": float(rt.sum() / camp.sum())}
for col in ("H_dst", "k_dst", "n"):
    z = resid(col)
    a, b = z[rt], z[camp & ~rt]
    res = {"mw_resid_mean_rt": float(np.nanmean(a)), "mw_resid_mean_other_campaign": float(np.nanmean(b)),
           "mw_p_greater": float(mannwhitneyu(a[~np.isnan(a)], b[~np.isnan(b)], alternative="greater").pvalue),
           "auc_raw": auc_rt(h[col].to_numpy(dtype=float)), "auc_cell_mean_only": auc_rt(cell_mean(col)),
           "auc_resid": auc_rt(z)}
    for kind, x in (("LEVEL", z), ("CUSUM", z), ("RATE", rate_stat(z))):
        lo, hi = 0.0, 100.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            al = alarms(x, kind, mid)
            lo, hi = (mid, hi) if np.sum(~rt[al]) / free_days > TARGET else (lo, mid)
        al = alarms(x, kind, hi)
        n_rt, n_free_camp = int(rt[al].sum()), int((camp & ~rt)[al].sum())
        p_hit = n_rt / max(len(al), 1)
        res[kind] = {"threshold": hi, "alarms_total": int(len(al)), "alarms_in_rt_hours": n_rt,
                     "alarms_in_other_campaign_hours": n_free_camp,
                     "alarms_per_day_free_hours": float(np.sum(~rt[al]) / free_days),
                     "p_rt_share_vs_hour_share": float(binomtest(n_rt, max(len(al), 1),
                                                                rt.mean(), alternative="greater").pvalue)}
    out[col] = res
# How much could red-team logons move the hourly counts themselves?
sd_n = h[~rt].groupby(["hod", "dow"])["n"].std()
cell_sd = h.set_index(["hod", "dow"]).index.map(sd_n).to_numpy()
ratio = h.rt_events.to_numpy()[rt] / cell_sd[rt]
out["count_shift_by_redteam"] = {"max_rt_events_per_hour": int(h.rt_events.max()),
                                 "median_cell_sd_of_hourly_count": float(np.nanmedian(cell_sd)),
                                 "max_rt_events_over_cell_sd": float(np.nanmax(ratio)),
                                 "median_rt_events_over_cell_sd": float(np.nanmedian(ratio))}
dev = pd.Series(h["H_dst"].to_numpy() - cell_mean("H_dst"))
out["H_dst_cell_residual"] = {"sd_bits": float(dev.std()),
                              **{f"autocorr_lag{L}h": float(dev.autocorr(L)) for L in (1, 6, 24)}}
json.dump(out, open(os.path.join(RES, "hourly_checks.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
