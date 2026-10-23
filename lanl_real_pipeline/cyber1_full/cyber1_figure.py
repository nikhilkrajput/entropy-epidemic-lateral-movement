"""
Figure for the LANL red-team evaluation (article Fig. 4):
  (a) hourly destination-host entropy over 58 days, campaign shaded, hours
      with red-team events marked (results/hourly.csv);
  (b) for each per-host score, the cumulative distribution of the within-day
      rank (as a fraction of the day's host-days, best first) of the 29
      red-team host-days (results/hostday_perhost.csv.gz).  A random score
      follows the diagonal; the area above it equals AUC - 0.5.
Writes ../../figures/fig_cyber1_full.(pdf|png).
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "..", "..", "figures")
ONSET, END = 1.7463541666666667, 29.59545138888889

h = pd.read_csv(os.path.join(RES, "hourly.csv"))
hd = pd.read_csv(os.path.join(RES, "hostday_perhost.csv.gz"))
rt = h.rt_events > 0

fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [2.1, 1]})
a = ax[0]
a.axvspan(ONSET, END, color="0.92", label="red-team campaign")
a.plot(h.day, h.H_dst, color="0.35", lw=0.6, label="hourly $H^{dst}$")
a.scatter(h.day[rt], h.H_dst[rt], s=14, color="tab:red", zorder=3,
          label=f"hours with red-team events ({int(rt.sum())})")
a.set_xlabel("Day"); a.set_ylabel("Destination-host entropy (bits)")
a.set_title("(a) Network-wide hourly entropy")
a.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False); a.grid(alpha=0.3)

b = ax[1]
first, last = int(hd.loc[hd.label == 1, "day"].min()), int(hd.loc[hd.label == 1, "day"].max())
c = hd[(hd.day >= first) & (hd.day <= last)].copy()
scores = {"X_host": ("cross-entropy vs. own history", "tab:green"),
          "new_dst": ("new destinations", "tab:blue"),
          "H_today": ("destination entropy", "tab:purple"),
          "n": ("logon count", "0.5")}
for sc, (label, col) in scores.items():
    frac = c.groupby("day")[sc].rank(ascending=False, pct=True)
    x = np.sort(frac[c.label == 1].to_numpy())
    b.step(np.concatenate([[0], x, [1]]), np.concatenate([[0], np.arange(1, len(x) + 1) / len(x), [1]]),
           where="post", color=col, lw=1.6, label=label)
xr = np.logspace(-4, 0, 200)
b.plot(xr, xr, "k:", lw=1, label="random")
b.set_xscale("log"); b.set_xlim(1e-4, 1)
b.set_xlabel("Within-day rank (fraction of host-days)")
b.set_ylabel("Share of red-team host-days")
b.set_title("(b) Per-host scores")
b.legend(fontsize=7, loc="upper left"); b.grid(alpha=0.3)
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(os.path.join(FIG, f"fig4_cyber1.{ext}"), bbox_inches="tight", dpi=300)
print("saved")
