"""
Additional checks on the cached 58-day LANL window features.

1. Mechanical coupling: H_S^dst is bounded by log2(number of distinct
   destinations k).  Report corr(H_dst, log2 k), the share of the bound
   reached (H_dst / log2 k), and the partial correlation of H_dst with
   iota_W after controlling for log2 k.
2. Prediction of iota_W WITHOUT destination entropy: source/user/spectral
   entropies only, linear regression and Random Forest, 10-fold
   day-blocked CV (same folds as enhanced_analysis.py).
3. Trend: correlation of each entropy with calendar day, to show that the
   cumulative-reach correlations reduce to a 58-day trend.

Writes outputs/run_58day_full/extra_checks.json.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs", "run_58day_full")

feat = pd.read_csv(os.path.join(OUT, "lanl_features.csv"))
N = json.load(open(os.path.join(OUT, "lanl_results.json")))["n_hosts_network"]
k = np.rint(feat["iota"].to_numpy() * N)
H = feat["H_S_dst"].to_numpy()
iota = feat["iota"].to_numpy()
logk = np.log2(np.maximum(k, 1))

res = {"distinct_dst_per_window": {"min": int(k.min()), "median": float(np.median(k)),
                                   "max": int(k.max())}}
res["corr_Hdst_log2k"] = float(pearsonr(H, logk)[0])
frac = H[k > 1] / logk[k > 1]
res["Hdst_over_log2k"] = {"median": float(np.median(frac)), "p10": float(np.percentile(frac, 10)),
                          "p90": float(np.percentile(frac, 90))}
# partial correlation of H with iota given log2 k
rH = H - np.polyval(np.polyfit(logk, H, 1), logk)
rI = iota - np.polyval(np.polyfit(logk, iota, 1), logk)
res["partial_corr_Hdst_iota_given_log2k"] = float(pearsonr(rH, rI)[0])

groups = feat["t_day"].astype(int).to_numpy()
cv = GroupKFold(n_splits=10)
for name, cols in {"src_usr_spectral": ["H_S_src", "H_S_usr", "H_R_src", "H_T_src", "H_Sp"],
                   "all_six": ["H_S_src", "H_S_dst", "H_S_usr", "H_R_src", "H_T_src", "H_Sp"]}.items():
    X = feat[cols].to_numpy()
    for mname, m in {"linear": LinearRegression(),
                     "rf": RandomForestRegressor(n_estimators=200, max_depth=12,
                                                 n_jobs=-1, random_state=42)}.items():
        p = cross_val_predict(m, X, iota, cv=cv, groups=groups)
        res[f"dayblock_R2_{name}_{mname}"] = float(r2_score(iota, p))

X1 = feat[["H_S_dst"]].to_numpy()
res["dayblock_R2_Hdst_only_linear"] = float(r2_score(
    iota, cross_val_predict(LinearRegression(), X1, iota, cv=cv, groups=groups)))
ok = k > 1
res["corr_evenness_iota"] = float(pearsonr(H[ok] / logk[ok], iota[ok])[0])

# Next non-overlapping window (two strides ahead, same day) as the target:
# does entropy carry information about upcoming activity beyond the
# current window's own breadth?
day = feat["t_day"].astype(int).to_numpy()
nxt = np.arange(len(feat)) + 2
valid = nxt < len(feat)
valid[valid] &= day[nxt[valid]] == day[valid]
idx, y_next = np.flatnonzero(valid), iota[nxt[valid]]
g2 = day[idx]
sets = {"persistence_current_iota": iota[idx][:, None],
        "Hdst_only": feat["H_S_dst"].to_numpy()[idx][:, None],
        "all_six": feat[["H_S_src", "H_S_dst", "H_S_usr", "H_R_src", "H_T_src", "H_Sp"]].to_numpy()[idx],
        "all_six_plus_current_iota": np.column_stack([
            feat[["H_S_src", "H_S_dst", "H_S_usr", "H_R_src", "H_T_src", "H_Sp"]].to_numpy()[idx],
            iota[idx]])}
res["next_window_dayblock_R2_linear"] = {
    name: float(r2_score(y_next, cross_val_predict(LinearRegression(), X, y_next, cv=cv, groups=g2)))
    for name, X in sets.items()}

span = feat.groupby("t_day")["t_sec"].agg(lambda s: (s.max() - s.min()) / 60.0)
res["daily_prefix_span_minutes"] = {"median": float(span.median()), "min": float(span.min()),
                                    "max": float(span.max()), "total_hours": float(span.sum() / 60)}
res["window_spacing_seconds_median"] = float(np.median(np.diff(feat["t_sec"].to_numpy())[
    np.diff(feat["t_sec"].to_numpy()) > 0]))

res["corr_with_day"] = {c: float(pearsonr(feat["t_day"], feat[c])[0])
                        for c in ["H_S_dst", "H_S_src", "H_S_usr", "H_R_src", "H_T_src", "H_Sp", "iota_cum"]}
with open(os.path.join(OUT, "extra_checks.json"), "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=1))
