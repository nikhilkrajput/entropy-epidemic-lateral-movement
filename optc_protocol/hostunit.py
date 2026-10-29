"""
Per-(source, unit) scores from a table of (s, unit, d, count): the same
formulas as lanl_real_pipeline/cyber1_full/cyber1_perhost.py, for any time
unit.  All history terms are causal (units before the current one).
"""
import numpy as np
import pandas as pd


def host_unit_scores(sd, unit, K, alpha=1.0):
    sd = sd.sort_values(["s", "d", unit]).reset_index(drop=True)
    sd["prior_sd"] = sd.groupby(["s", "d"])["count"].cumsum() - sd["count"]
    tot = sd.groupby(["s", unit])["count"].sum().rename("n").reset_index().sort_values(["s", unit])
    tot["prior_s"] = tot.groupby("s")["n"].cumsum() - tot["n"]
    sd = sd.merge(tot, on=["s", unit])
    p_today = sd["count"] / sd["n"]
    sd["surp_host"] = -sd["count"] * np.log2((sd["prior_sd"] + alpha) / (sd["prior_s"] + alpha * K))
    sd["ent_today"] = -p_today * np.log2(p_today)
    sd["new"] = (sd["prior_sd"] == 0).astype(int)
    g = sd.groupby(["s", unit])
    hd = pd.DataFrame({"n": g["count"].sum(), "S_host": g["surp_host"].sum(), "H_today": g["ent_today"].sum(),
                       "new_dst": g["new"].sum(), "k_dst": g["d"].nunique()}).reset_index()
    hd["X_host"] = hd["S_host"] / hd["n"]
    hd["KL_host"] = hd["X_host"] - hd["H_today"]
    return hd
