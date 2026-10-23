"""
Upper bound on how much red-team logons can move hourly destination-host
entropy.  Replacing a share eps of the events changes the empirical
distribution by total variation <= eps, so by the Fannes-Audenaert
continuity bound |dH| <= eps*log2(K-1) + h2(eps), with K the number of
possible destinations (17,684 computers in cyber1) and h2 the binary
entropy.  Compared with the s.d. of attack-free hourly entropy in the
same hour-of-day x day-of-week cell.
Writes entropy_bound.json next to hourly.csv.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
K = 17684
h = pd.read_csv(os.path.join(HERE, "hourly.csv"))
h["hod"], h["dow"] = h.hour % 24, (h.hour // 24) % 7
rt = h.rt_events > 0
eps = (h.rt_events / h.n).where(rt)


def h2(p):
    p = np.clip(p, 1e-300, 1 - 1e-16)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


bound = eps * np.log2(K - 1) + h2(eps)
sd_cell = h[~rt].groupby(["hod", "dow"]).H_dst.std()
sd = h.set_index(["hod", "dow"]).index.map(sd_cell).to_numpy()
ratio = (bound / sd)[rt]
out = {"K": K, "n_redteam_hours": int(rt.sum()),
       "eps_median": float(eps[rt].median()), "eps_max": float(eps[rt].max()),
       "bound_bits_median": float(bound[rt].median()), "bound_bits_max": float(bound[rt].max()),
       "hourly_sd_bits_median": float(np.nanmedian(sd)),
       "bound_over_sd_median": float(np.nanmedian(ratio)), "bound_over_sd_max": float(np.nanmax(ratio))}
json.dump(out, open(os.path.join(HERE, "entropy_bound.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
