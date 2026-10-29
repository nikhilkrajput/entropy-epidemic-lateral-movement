"""
Fixed-budget evaluation of per-host-day scores, shared by the LANL and
OpTC analyses.

Input: a host-day table with one row per (source host s, day) and columns
  s, day, n (logons or flows that day), new_dst (destinations never
  contacted by s before today), label (1 if s was a labelled attack source
  that day), events (labelled attack events of s that day; optional), and
  any score columns (higher = more suspicious).

Alert rule: each day the analyst receives the day's b highest-scoring
host-days.  Ties are broken uniformly at random; we report the exact
expectation over tie-breaks, so results do not depend on row order: a
positive with g host-days strictly above it and e host-days with an equal
score (itself included) is alerted with probability clip((b - g) / e, 0, 1).

Metrics per score and budget b:
  recall      expected share of positive host-days alerted;
  precision   expected share of alerts that are positive host-days;
  sources     source hosts with at least one positive host-day alerted with
              probability >= 0.5, and the delay in days from the source's
              first positive host-day to that alert;
  event_cover_after_alert_day
              share of labelled attack events on days after their source's
              first alert day (an alert is available when its day closes:
              the part of the campaign an analyst could have interrupted);
  recall_with_history / recall_without_history
              recall over positives whose source has / has no earlier logons;
  ci95        95% interval of recall from a cluster bootstrap that
              resamples source hosts (positives of one source are not
              independent).
"""
import numpy as np
import pandas as pd
from scipy.stats import binom

BUDGETS = (1, 10, 100)
N_BOOT = 2000


def add_history(hd):
    """Causal per-host history columns (all days before today)."""
    hd = hd.sort_values(["s", "day"]).reset_index(drop=True)
    g = hd.groupby("s")
    hd["hist_n"] = g["n"].cumsum() - hd["n"]
    hd["hist_dst"] = g["new_dst"].cumsum() - hd["new_dst"]      # distinct destinations seen before today
    hd["age"] = hd["day"] - g["day"].transform("min")
    return hd


def novelty_baselines(hd):
    """host_age and -log history rank new hosts first; new_host_k is the
    stronger control: a host with no history scores its count of first
    contacts, every other host 0 (so new hosts are ranked by how many hosts
    they reach, as a new-host-plus-first-contact alarm would)."""
    hd["host_age"] = -hd["age"].astype(float)
    hd["neg_log_hist"] = -np.log1p(hd["hist_n"])
    hd["new_host_k"] = np.where(hd["hist_n"] == 0, hd["new_dst"], 0).astype(float)
    return hd


def population_novelty_rate(hd):
    """Causal network-wide rate at which logons go to a destination new for their source."""
    dd = hd.groupby("day")[["new_dst", "n"]].sum().sort_index()
    cum = dd.cumsum() - dd
    return ((cum["new_dst"] + 1) / (cum["n"] + 2)).rename("q_pop")


def eb_novelty(hd, a):
    """Empirical-Bayes binomial novelty surprise, in nats.

    A source host's chance that a logon goes to a destination it has never
    used is estimated by shrinking its own history (hist_dst new
    destinations in hist_n logons) towards the network-wide rate q_pop with
    a prior weight of a logons:  q_s = (hist_dst + a q_pop) / (hist_n + a).
    The score is -ln P(Binomial(n, q_s) >= new_dst): how improbable today's
    count of first contacts is for this host.  A new host is judged by the
    network's rate, a host that habitually contacts new machines by its own.
    """
    q_pop = hd["day"].map(population_novelty_rate(hd))
    q = q_pop if np.isinf(a) else (hd["hist_dst"] + a * q_pop) / (hd["hist_n"] + a)
    return -binom.logsf(hd["new_dst"] - 1, hd["n"], q)


def alert_prob(c, score, b):
    """Probability that each row is among its day's top-b, ties broken at random.
    A missing score ranks lowest; +inf ranks highest (ties among them at random)."""
    s = np.nan_to_num(c[score].to_numpy(float), nan=-np.inf, posinf=np.inf)
    day = c["day"].to_numpy()
    p = np.zeros(len(c))
    for d in np.unique(day):
        idx = np.flatnonzero(day == d)
        v = s[idx]
        order = np.sort(v)[::-1]
        g = len(v) - np.searchsorted(order[::-1], v, side="right")      # strictly greater
        e = (len(v) - np.searchsorted(order[::-1], v, side="left")) - g  # equal, incl. self
        p[idx] = np.clip((b - g) / e, 0.0, 1.0)
    return p


def evaluate(c, scores, budgets=BUDGETS, n_boot=N_BOOT, seed=0):
    """c: evaluation host-days (history columns already computed)."""
    lab = c["label"].to_numpy() == 1
    src_all = c["s"].to_numpy()
    srcs = sorted(np.unique(src_all[lab]))
    ev = c["events"].to_numpy(float) if "events" in c else lab.astype(float)
    out = {"host_days": int(len(c)), "positives": int(lab.sum()), "sources": [int(x) for x in srcs],
           "positives_per_source": {int(s): int((lab & (src_all == s)).sum()) for s in srcs},
           "days": int(c["day"].nunique()), "median_host_days_per_day": float(c.groupby("day").size().median())}
    rng = np.random.default_rng(seed)
    boots = [rng.choice(len(srcs), len(srcs), replace=True) for _ in range(n_boot)]
    for sc in scores:
        r = {}
        for b in budgets:
            p = alert_prob(c, sc, b)
            pos_p = p[lab]
            pos_src = src_all[lab]
            per_src = [pos_p[pos_src == s] for s in srcs]
            bs = []
            for pick in boots:
                v = np.concatenate([per_src[i] for i in pick])
                bs.append(v.mean())
            # An alert on any unit of day t is available when day t closes, so
            # delay is in days and coverage counts the labelled events of later
            # days only: the part of the campaign an analyst could interrupt.
            det, delays, covered = [], [], 0.0
            cd = c.loc[lab, ["s", "day"]].assign(p=pos_p, ev=ev[lab])
            for s in srcs:
                x = cd[cd.s == s].sort_values("day")
                hit = x[x.p >= 0.5]
                if len(hit):
                    det.append(int(s))
                    delays.append(float(hit.day.iloc[0] - x.day.iloc[0]))
                    covered += x.loc[x.day > hit.day.iloc[0], "ev"].sum()
            hist = c["hist_n"].to_numpy()[lab] > 0 if "hist_n" in c else np.zeros(lab.sum(), bool)
            r[f"b{b}"] = {"recall": float(pos_p.mean()),
                          "recall_ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                          "recall_per_source": {int(s): float(v.mean()) for s, v in zip(srcs, per_src)},
                          "recall_with_history": float(pos_p[hist].mean()) if hist.any() else None,
                          "recall_without_history": float(pos_p[~hist].mean()) if (~hist).any() else None,
                          "positives_with_history": int(hist.sum()),
                          "precision": float(pos_p.sum() / (b * c["day"].nunique())),
                          "sources_detected": det, "n_sources_detected": len(det),
                          "detection_delay_days": delays,
                          "event_cover_after_alert_day": float(covered / ev[lab].sum())}
        out[sc] = r
    return out


def threshold_rule(hd, score, b, window=7, eval_days=None):
    """SIEM-style rule: alert on a host-unit whose score reaches tau_t, the
    median over the previous `window` days of each day's b-th largest score.
    The threshold uses past days only; days without a past day get no alerts.
    Returns realised alerts per day and recall / sources on eval_days."""
    kth = hd.groupby("day")[score].apply(lambda v: np.sort(v.to_numpy())[::-1][min(b, len(v)) - 1]).sort_index()
    tau = kth.shift(1).rolling(window, min_periods=1).median()
    t = hd["day"].map(tau)
    fl = (hd[score] >= t).to_numpy()
    m = np.ones(len(hd), bool) if eval_days is None else hd["day"].between(*eval_days).to_numpy()
    lab = hd["label"].to_numpy() == 1
    per_day = pd.Series(fl[m]).groupby(hd["day"].to_numpy()[m]).sum()
    det = sorted(int(s) for s in hd.loc[fl & m & lab, "s"].unique())
    return {"b": b, "window_days": window, "alerts_per_day_mean": float(per_day.mean()),
            "alerts_per_day_median": float(per_day.median()), "alerts_per_day_max": int(per_day.max()),
            "recall": float(fl[m & lab].mean()), "sources_detected": det, "n_sources_detected": len(det)}


def within_day_auc(c, scores):
    from sklearn.metrics import roc_auc_score
    return {sc: float(roc_auc_score(c["label"], c.groupby("day")[sc].rank(pct=True))) for sc in scores}
