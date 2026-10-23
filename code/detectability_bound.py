"""
Detectability threshold for network-wide entropy monitors.

If a share eps of a window's events is attack traffic, the empirical
distribution is within total variation eps of the same window with those
events replaced by benign ones, and by the sharp continuity bound for
Shannon entropy (Zhang 2007; Audenaert 2007)
    |dH| <= B(eps) = eps * log2(K - 1) + h2(eps),   eps <= 1 - 1/K.
A monitor that needs a shift of z baseline standard deviations therefore
cannot respond unless B(eps) >= z * sigma; eps*(z) solves B(eps) = z*sigma.

Sequential detectors can accumulate small shifts, so the bound is also
turned into a detection-delay limit.  If attack-free windows are
independent N(mu, sigma^2) and the attack moves the mean by at most B, the
Kullback-Leibler information per window is at most B^2 / (2 sigma^2), and
for any sequential detector whose mean time to false alarm is at least
gamma windows the worst-case mean delay is at least
    ln(gamma) * 2 sigma^2 / B^2       (Lorden 1971; Lai 1998),
asymptotically as gamma grows.  Over a finite campaign, the expected
log-likelihood ratio a detector can collect is at most
sum_t B_t^2 / (2 sigma_t^2), which is compared with ln(gamma).

Computes eps* for the LANL red-team data (hourly windows) and for the
simulator, and checks the bound against the entropy shifts the simulator
actually produces.  Writes data/detectability_bound.json.
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import traffic_sim as TS  # noqa: E402

ROOT = os.path.join(HERE, "..")


def h2(p):
    p = np.clip(p, 1e-300, 1 - 1e-16)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def B(eps, K):
    return eps * np.log2(K - 1) + h2(eps)


def eps_star(z, sigma, K):
    return brentq(lambda e: B(e, K) - z * sigma, 1e-12, 1 - 1.0 / K)


out = {}
# --- LANL red-team data, hourly destination entropy
eb = json.load(open(os.path.join(ROOT, "lanl_real_pipeline", "cyber1_full", "results", "entropy_bound.json")))
import pandas as pd  # noqa: E402
hourly = pd.read_csv(os.path.join(ROOT, "lanl_real_pipeline", "cyber1_full", "results", "hourly.csv"))
K, sigma = eb["K"], eb["hourly_sd_bits_median"]
n_med = float(hourly.n.median())
out["lanl_cyber1_hourly"] = {
    "K": K, "sigma_bits": sigma, "median_events_per_hour": n_med,
    **{f"z{z}": {"eps_star": eps_star(z, sigma, K),
                 "malicious_logons_per_hour": eps_star(z, sigma, K) * n_med}
       for z in (1, 2, 3)},
    "campaign_eps_max": eb["eps_max"], "campaign_eps_median": eb["eps_median"],
}

# --- detection-delay limit (hourly windows, 0.01 false alarms per day)
gamma_h = 24 / 0.01
h = hourly.copy()
h["hod"], h["dow"] = h.hour % 24, (h.hour // 24) % 7
rt = h.rt_events > 0
sd_cell = h[~rt].groupby(["hod", "dow"]).H_dst.std()
sd_t = h[rt].set_index(["hod", "dow"]).index.map(sd_cell).to_numpy()
B_t = B((h.rt_events / h.n)[rt].to_numpy(), K)
evidence = float(np.nansum(B_t ** 2 / (2 * sd_t ** 2)))
out["lanl_cyber1_hourly"]["delay_limit"] = {
    "mean_time_to_false_alarm_hours": gamma_h, "ln_gamma": float(np.log(gamma_h)),
    "min_delay_hours_at_max_bound": float(np.log(gamma_h) * 2 * sigma ** 2 / eb["bound_bits_max"] ** 2),
    "min_delay_hours_at_median_bound": float(np.log(gamma_h) * 2 * sigma ** 2 / eb["bound_bits_median"] ** 2),
    "campaign_evidence_nats_upper": evidence,
    "campaign_evidence_over_ln_gamma": evidence / float(np.log(gamma_h)),
}

# --- Simulator: compare actual entropy shift with the bound
rng = np.random.default_rng(3)
net = TS.Network(rng)
base = net.windows(np.zeros(4000), rng)["H_dst"]
sim = {"K": TS.N_HOSTS, "sigma_bits": float(base.std()),
       "eps_star_z1": eps_star(1, float(base.std()), TS.N_HOSTS), "points": []}
for m in (0.5, 1.0, 2.0, 5.0):
    for I in (5, 10, 20, 50):
        o = net.windows(np.full(4000, I), rng, m=m)
        eps = float(o["mal_share"].mean())
        dH = float(o["H_dst"].mean() - base.mean())
        sim["points"].append({"m": m, "I": I, "eps": eps, "dH_bits": dH,
                              "bound_bits": float(B(eps, TS.N_HOSTS)),
                              "dH_over_bound": dH / float(B(eps, TS.N_HOSTS))})
out["simulator"] = sim
with open(os.path.join(ROOT, "data", "detectability_bound.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out["lanl_cyber1_hourly"], indent=1))
r = [p["dH_over_bound"] for p in sim["points"]]
print("simulator: sigma %.3f eps*(z=1) %.4f; dH/bound range %.2f-%.2f; any violation: %s"
      % (sim["sigma_bits"], sim["eps_star_z1"], min(r), max(r), any(x > 1 for x in r)))
