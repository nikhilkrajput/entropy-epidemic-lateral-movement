"""
Fine-window analysis of the LANL red-team data: does destination-host
entropy respond at the paper's window size (W = 500 consecutive logons)?

For consecutive non-overlapping windows of W events (Network and
RemoteInteractive logons, in log order) we record the destination-host
Shannon entropy, the number of red-team events and the window's hour.
Outputs:
  * red-team coverage: distinct (time, src, dst) triples in the labels and
    how many occur among the analysed events;
  * the Proposition-1 bound B_K(eps) for each window containing red-team
    events, relative to the s.d. of red-team-free windows in the same hour;
  * an empirical test: entropy of red-team windows standardised against
    the red-team-free windows of the same hour (mean z, ROC AUC).
Writes cyber1_windows_results.json and windows_rt.csv (windows with
red-team events plus per-hour baseline statistics).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
FILT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "filt.csv")
RT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "redteam.txt.gz")
W = 500
K = 17684
LT_KEEP = {"Network", "RemoteInteractive"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def h2(p):
    p = np.clip(p, 1e-300, 1 - 1e-16)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


rt = pd.read_csv(RT, header=None, names=["time", "user", "src", "dst"])
keys = rt[["time", "src", "dst"]].drop_duplicates()
keys["is_rt"] = 1
found = set()
win_rows = []          # per window: id, hour, H, n_rt
carry = None           # incomplete window carried across chunks
offset = 0
for ch in pd.read_csv(FILT, header=None, names=["time", "src", "dst", "lt"], chunksize=10_000_000,
                      dtype={"time": np.int64, "src": str, "dst": str, "lt": str}):
    ch = ch[ch["lt"].isin(LT_KEEP)][["time", "src", "dst"]]
    if carry is not None:
        ch = pd.concat([carry, ch], ignore_index=True)
    m = ch.merge(keys, on=["time", "src", "dst"], how="left")
    m["is_rt"] = m["is_rt"].fillna(0).astype(int)
    for t in m.loc[m.is_rt == 1, ["time", "src", "dst"]].itertuples(index=False):
        found.add(tuple(t))
    n_full = (len(m) // W) * W
    carry = ch.iloc[n_full:].copy()
    m = m.iloc[:n_full]
    wid = np.arange(n_full) // W + offset
    offset += n_full // W
    cnt = pd.DataFrame({"w": wid, "d": m["dst"].to_numpy()}).value_counts().rename("c").reset_index()
    p = cnt["c"] / W
    H = (-(p * np.log2(p))).groupby(cnt["w"]).sum()
    agg = pd.DataFrame({"w": wid, "t": m["time"].to_numpy(), "rt": m["is_rt"].to_numpy()}).groupby("w").agg(
        t0=("t", "min"), n_rt=("rt", "sum"))
    agg["H"] = H
    win_rows.append(agg.reset_index())
    log("windows", offset, "rt triples found", len(found))

win = pd.concat(win_rows, ignore_index=True)
win["hour"] = win["t0"] // 3600
base = win[win.n_rt == 0].groupby("hour")["H"].agg(["mean", "std", "size"])
win = win.join(base, on="hour")
win["z"] = (win["H"] - win["mean"]) / win["std"]
r = win[win.n_rt > 0].copy()
r["eps"] = r["n_rt"] / W
r["bound"] = r["eps"] * np.log2(K - 1) + h2(r["eps"])
r["bound_over_sd"] = r["bound"] / r["std"]
r.to_csv(os.path.join(HERE, "windows_rt.csv"), index=False)

# empirical: red-team windows vs red-team-free windows in the same hours
same_hours = win[win.hour.isin(r.hour.unique())]
lab = (same_hours.n_rt > 0).astype(int)
auc = float(roc_auc_score(lab, same_hours["z"].fillna(0)))
out = {
    "W": W, "n_windows": int(len(win)), "window_seconds_median": float(np.median(np.diff(win["t0"]))),
    "redteam_records": int(len(rt)), "redteam_distinct_triples": int(len(keys)),
    "redteam_triples_found": int(len(found)),
    "windows_with_redteam": int(len(r)), "redteam_events_in_windows": int(r.n_rt.sum()),
    "eps_median": float(r.eps.median()), "eps_max": float(r.eps.max()),
    "bound_bits_median": float(r.bound.median()), "bound_bits_max": float(r.bound.max()),
    "baseline_sd_bits_median": float(base["std"].median()),
    "bound_over_sd_median": float(r.bound_over_sd.median()),
    "bound_over_sd_max": float(r.bound_over_sd.max()),
    "frac_rt_windows_bound_over_sd_ge_1": float((r.bound_over_sd >= 1).mean()),
    "rt_window_z_mean": float(r.z.mean()), "rt_window_z_median": float(r.z.median()),
    "auc_rt_windows_vs_same_hour": auc,
    "rt_windows_with_z_ge_3": int((r.z >= 3).sum()),
}
json.dump(out, open(os.path.join(HERE, "cyber1_windows_results.json"), "w"), indent=2)
log(json.dumps(out))
