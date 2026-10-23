"""
Fig. 3 and the reach and linkage numbers of the LANL 2017 section, from the
cached outputs (no raw data needed):
  outputs/run_58day_full/{lanl_features.csv, lanl_results.json}

Writes outputs/run_58day_full/growth.json and figures/fig3_lanl_reach_linkage.*
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs", "run_58day_full")
PAPER_FIG = os.path.join(HERE, "..", "figures")
ENTROPY_COLS = ["H_S_dst", "H_S_src", "H_T_src", "H_R_src", "H_S_usr", "H_Sp"]

feat = pd.read_csv(os.path.join(OUT, "lanl_features.csv"))
res = json.load(open(os.path.join(OUT, "lanl_results.json")))
N = res["n_hosts_network"]

daily_iota = np.array(res["daily_aggregate"]["cum_iota"])
days = np.arange(1, len(daily_iota) + 1)
I = daily_iota * N

lin = np.polyfit(days, I, 1)
lin_pred = np.polyval(lin, days)
lin_r2 = 1.0 - np.sum((I - lin_pred) ** 2) / np.sum((I - I.mean()) ** 2)

nw = 7
r_ll = float(np.polyfit(days[:nw], np.log(I[:nw]), 1)[0])
growth_ratio = I[1:] / I[:-1]


corr = {}
for c in ENTROPY_COLS:
    rw, pw = pearsonr(feat["iota"], feat[c])
    rc, pc = pearsonr(feat["iota_cum"], feat[c])
    corr[c] = {"r_window": float(rw), "p_window": float(pw),
               "r_cumulative": float(rc), "p_cumulative": float(pc)}

summary = {
    "linear_fit": {"slope_hosts_per_day": float(lin[0]), "R2": float(lin_r2)},
    "first_day_hosts": float(I[0]), "last_day_hosts": float(I[-1]),
    "day_over_day_growth_ratio_first6": growth_ratio[:6].tolist(),
    "loglinear_first7d_r_per_day": r_ll,
    "median_window_iota": float(feat["iota"].median()),
    "correlations": corr,
}
with open(os.path.join(OUT, "growth.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps({k: v for k, v in summary.items() if k != "correlations"}, indent=1))
for c, v in corr.items():
    print(f"{c:8s} r_win={v['r_window']:+.3f} (p={v['p_window']:.1e})  "
          f"r_cum={v['r_cumulative']:+.3f} (p={v['p_cumulative']:.1e})")

# ---------------- Fig. 3: reach curve + linkage ----------------
iota = feat["iota"].to_numpy()
H = feat["H_S_dst"].to_numpy()
coef = np.polyfit(iota, H, 1)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
a = ax[0]
a.plot(days, I, "ko", ms=5, label="Cumulative distinct destination hosts")
a.plot(days, lin_pred, "b-", lw=1.8,
       label=f"Linear fit ($R^2={lin_r2:.3f}$, {lin[0]:.0f} hosts/day)")
a.set_xlabel("Day"); a.set_ylabel("Hosts reached")
a.set_title("(a) Cumulative reach, 58 daily snapshots")
a.legend(fontsize=8, loc="upper left"); a.grid(alpha=0.3)
a = ax[1]
a.scatter(iota, H, s=6, alpha=0.3, color="steelblue", label=f"Windows (n={len(feat):,})")
xf = np.linspace(iota.min(), iota.max(), 50)
a.plot(xf, np.polyval(coef, xf), "r-", lw=1.8,
       label=f"OLS: $H={coef[1]:.2f}+{coef[0]:.0f}\\,\\iota_W$")
a.set_xlabel("Per-window active-destination fraction $\\iota_W$")
a.set_ylabel("$H^{dst}$ (bits)")
a.set_title(f"(b) Per-window association, Pearson $r={corr['H_S_dst']['r_window']:.3f}$")
a.legend(fontsize=8); a.grid(alpha=0.3)
plt.tight_layout()
for ext in (".pdf", ".png"):
    plt.savefig(os.path.join(PAPER_FIG, "fig3_lanl_reach_linkage" + ext), bbox_inches="tight", dpi=300)
plt.close()
print("regenerated fig3_lanl_reach_linkage")
