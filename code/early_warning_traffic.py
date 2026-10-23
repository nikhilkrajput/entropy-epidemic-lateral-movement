"""
Early warning on simulated logon traffic (traffic_sim.py), with
false-alarm control.

Design
------
* 230 days, one 500-logon window every 6 hours; attack-free until day 30,
  then one seed host and a deterministic SIR epidemic (beta ~ U(0.15, 0.55),
  gamma ~ U(0.04, 0.20)).  Onset = first time iota > 0.01.
* The monitored observable is destination-host Shannon entropy.
* Detectors (causal; mu0, sd0 from days 0-20 of the run):
    LEVEL   z_t > h
    CUSUM   g_t = max(0, g_{t-1} + z_t - 0.5) > h
    CUSUM2  as CUSUM with reference k = 0.2 (tuned for small shifts)
    RATE    trailing 4-day least-squares slope / its calibration s.d. > h
  Each detector runs continuously from day 20 (it is not told the seed
  time); after an alarm it is reset and silenced for 4 days.  Lead time
  uses the first alarm at or after the seed; alarms before the seed are
  counted as false alarms.
* Thresholds: bisection to 0.01 false alarms/day on N_CAL attack-free
  series per baseline setting; the realised rate is re-measured on N_CAL
  fresh series and used for the chance level.
* Conditions: lateral intensity m in {0.5, 1, 2, 5} and three baseline
  settings: none, independent variability (s_a = 0.15) and autocorrelated
  variability (s_a = 0.15, AR(1) rho = 0.63, which gives destination
  entropy a lag-1 correlation of about 0.57 as in LANL 2017).  Scenarios
  and benign traffic are identical across m (traffic_sim pairs the
  benign stream).

Outputs: data/early_warning_traffic.json, figures/fig2_early_warning_snr.(pdf|png)
"""
import json
import os
import sys

import numpy as np
from scipy.integrate import odeint
from scipy.stats import binomtest
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import traffic_sim as TS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
FIG_DIR = os.path.join(HERE, "..", "figures")

N = TS.N_HOSTS
DT, T_ATTACK, T_END, CAL_DAYS = 0.25, 30.0, 230.0, 20.0
RW = 16
K_REF = {"CUSUM": 0.5, "CUSUM2": 0.2}
TARGET_FAR = 0.01
N_RUNS = 200
N_CAL = 1000
N_SNR = 4000
DETECTORS = ("LEVEL", "CUSUM", "CUSUM2", "RATE")
MS = (0.5, 1.0, 2.0, 5.0)
BASELINES = {"none": (0.0, 0.0), "iid": (0.15, 0.0), "ar1": (0.15, 0.63)}

t = np.arange(0.0, T_END, DT)
n_cal = int(CAL_DAYS / DT)
i_att = int(T_ATTACK / DT)
_x = (np.arange(RW) - (RW - 1) / 2.0) * DT
_wslope = _x / np.sum(_x ** 2)


def sir(y, _t, b, g):
    S, I, R = y
    return [-b * S * I / N, b * S * I / N - g * I, g * I]


def stats(H):
    mu, sd = H[:n_cal].mean(), H[:n_cal].std(ddof=1)
    z = (H - mu) / sd
    sl = np.full(len(H), np.nan)
    sl[RW - 1:] = np.convolve(H, _wslope[::-1], mode="valid")
    return {"LEVEL": z, "CUSUM": z, "CUSUM2": z, "RATE": sl / np.nanstd(sl[:n_cal])}


def alarms(x, kind, h, start):
    """Alarm indices from `start` with reset and refractory period RW."""
    out = []
    if kind in K_REF:
        k = K_REF[kind]
        pos = start
        while pos < len(x):
            S = np.cumsum(x[pos:] - k)
            g = S - np.minimum(0.0, np.minimum.accumulate(S))
            hit = np.flatnonzero(g > h)
            if hit.size == 0:
                break
            a = pos + int(hit[0]); out.append(a); pos = a + RW
        return out
    exc = np.flatnonzero(np.nan_to_num(x[start:], nan=-np.inf) > h) + start
    nxt = start
    for e in exc:
        if e >= nxt:
            out.append(int(e)); nxt = e + RW
    return out


def far(series, kind, h):
    return sum(len(alarms(s, kind, h, n_cal)) for s in series) / (len(series) * (T_END - CAL_DAYS))


def attack_free(n_series, s_a, rho, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_series):
        net = TS.Network(rng)
        out.append(stats(net.windows(np.zeros(len(t)), rng, s_a=s_a, rho=rho)["H_dst"]))
    return out


def calibrate(name, s_a, rho):
    cal = attack_free(N_CAL, s_a, rho, seed=7)
    fresh = attack_free(N_CAL, s_a, rho, seed=8)
    res = {}
    for k in DETECTORS:
        cs = [c[k] for c in cal]
        lo, hi = 0.0, 80.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if far(cs, k, mid) > TARGET_FAR else (lo, mid)
        res[k] = {"h": hi, "far_cal": far(cs, k, hi), "far_fresh": far([c[k] for c in fresh], k, hi)}
        print(f"  {name} {k}: h={hi:.3f} far cal={res[k]['far_cal']:.4f} fresh={res[k]['far_fresh']:.4f}",
              flush=True)
    return res


def make_runs():
    rng = np.random.default_rng(2026)
    runs = []
    for i in range(N_RUNS):
        b, g = rng.uniform(0.15, 0.55), rng.uniform(0.04, 0.20)
        iota = np.zeros(len(t))
        iota[i_att:] = odeint(sir, [N - 1, 1, 0], t[i_att:] - T_ATTACK, args=(b, g))[:, 1] / N
        runs.append({"iota": iota, "seed": 50_000 + i})
    return runs


def evaluate(runs, m, s_a, rho, thr):
    res = {k: {"lead": [], "missed": 0, "pre": 0, "n_pre": 0} for k in DETECTORS}
    delays = []
    for r in runs:
        iota = r["iota"]
        if not (iota > 0.01).any():
            continue
        rng = np.random.default_rng(r["seed"])
        net = TS.Network(rng)
        H = net.windows(np.rint(iota * N), rng, m=m, s_a=s_a, rho=rho)["H_dst"]
        onset = t[np.argmax(iota > 0.01)]
        delays.append(onset - T_ATTACK)
        st = stats(H)
        for k in DETECTORS:
            al = alarms(st[k], k, thr[k]["h"], n_cal)        # one continuous run
            n_pre = sum(i < i_att for i in al)
            res[k]["pre"] += n_pre > 0; res[k]["n_pre"] += n_pre
            post = [i for i in al if i >= i_att]
            if post:
                res[k]["lead"].append(onset - t[post[0]])
            else:
                res[k]["missed"] += 1
    delays = np.array(delays); n = len(delays)
    out = {"n_runs_with_onset": n, "mean_onset_after_attack_days": float(delays.mean())}
    for k in DETECTORS:
        L = np.array(res[k]["lead"])
        cens = np.concatenate([L, np.full(res[k]["missed"], -np.inf)])
        chance = float(np.mean(1 - np.exp(-thr[k]["far_fresh"] * delays)))
        ne = int(np.sum(L > 0))
        out[k] = {"lead_median_censored": float(np.median(cens)),
                  "lead_median_detected": float(np.median(L)) if len(L) else None,
                  "missed": res[k]["missed"], "early_fraction": ne / n,
                  "chance_early_fraction": chance,
                  "early_vs_chance_p": float(binomtest(ne, n, chance, alternative="greater").pvalue),
                  "pre_attack_false_alarm_runs": res[k]["pre"],
                  "pre_attack_false_alarms_per_day": res[k]["n_pre"] / (n * (T_ATTACK - CAL_DAYS))}
    return out, res


def onset_snr(m, s_a, rho):
    """Mean entropy shift at iota = 0.01 over attack-free s.d., with s.e."""
    r = np.random.default_rng(7)
    net = TS.Network(r)
    base = net.windows(np.zeros(N_SNR), r, m=m, s_a=s_a, rho=rho)["H_dst"]
    on = net.windows(np.full(N_SNR, 10), r, m=m, s_a=s_a, rho=rho)["H_dst"]
    sd = base.std()
    snr = float((on.mean() - base.mean()) / sd)
    se = float(np.sqrt(on.var() / N_SNR + base.var() / N_SNR) / sd)
    return snr, se


def plot(out):
    """Fig. 2 from the saved results (data/early_warning_traffic.json)."""
    cols = {"LEVEL": "tab:blue", "CUSUM": "tab:green", "CUSUM2": "tab:olive", "RATE": "tab:purple"}
    styles = {"none": "-", "iid": "--", "ar1": ":"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for name in BASELINES:
        x = [out[f"m{m}_{name}"]["snr_on"] for m in MS]
        for k in DETECTORS:
            y = [max(out[f"m{m}_{name}"][k]["lead_median_censored"], -30) for m in MS]
            ax[0].plot(x, y, styles[name], marker="o", ms=4, color=cols[k])
            ax[1].plot(x, [out[f"m{m}_{name}"][k]["early_fraction"] for m in MS], styles[name],
                       marker="o", ms=4, color=cols[k])
    chance = [out[f"m{m}_{n}"][k]["chance_early_fraction"] for m in MS for n in BASELINES for k in DETECTORS]
    ax[1].axhspan(min(chance), max(chance), color="0.85", label="chance range")
    ax[0].axhline(0, color="k", lw=0.8)
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("Onset SNR"); a.grid(alpha=0.3)
        a.set_xticks([0.1, 0.2, 0.5, 1, 2, 5])
        a.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5"])
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].set_ylabel("Median lead vs. onset (days; clipped at -30)")
    ax[0].set_title("(a) Detection timing")
    ax[1].set_ylabel("Fraction of runs alarming before onset")
    ax[1].set_title("(b) Early alarms")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, marker="o", label=k.lower().replace("cusum2", "cusum, k=0.2"))
               for k, c in cols.items()]
    bname = {"none": "no variability", "iid": "independent", "ar1": "AR(1)"}
    handles += [Line2D([], [], color="k", ls=s, label=f"baseline: {bname[n]}") for n, s in styles.items()]
    ax[1].legend(handles=handles + [plt.Rectangle((0, 0), 1, 1, color="0.85", label="chance range")],
                 fontsize=7, ncol=2, loc="upper left")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG_DIR, f"fig2_early_warning_snr.{ext}"), bbox_inches="tight", dpi=300)
    plt.close()
    print("saved")


def main():
    runs = make_runs()
    out = {"settings": {"dt_days": DT, "attack_day": T_ATTACK, "horizon_days": T_END,
                        "calibration_days": CAL_DAYS, "rate_window_refractory_samples": RW,
                        "cusum_k": K_REF, "target_far_per_day": TARGET_FAR,
                        "n_runs": N_RUNS, "n_calibration_series": N_CAL,
                        "snr_windows_each": N_SNR, "baselines": BASELINES,
                        "observable": "destination-host Shannon entropy", "onset": "iota > 0.01",
                        "detector_run": "continuous from the end of calibration; not reset at the seed"}}
    thr_all = {}
    for name, (s_a, rho) in BASELINES.items():
        print("calibrating", name, flush=True)
        thr_all[name] = calibrate(name, s_a, rho)
        out[f"thresholds_{name}"] = thr_all[name]
    for name, (s_a, rho) in BASELINES.items():
        for m in MS:
            res, _ = evaluate(runs, m, s_a, rho, thr_all[name])
            res["snr_on"], res["snr_on_se"] = onset_snr(m, s_a, rho)
            out[f"m{m}_{name}"] = res
            print(f"m={m} {name} snr_on={res['snr_on']:.2f}",
                  {k: (round(res[k]["lead_median_censored"], 2), res[k]["missed"],
                       round(res[k]["early_fraction"], 3), round(res[k]["early_vs_chance_p"], 4))
                   for k in DETECTORS}, flush=True)
    with open(os.path.join(DATA_DIR, "early_warning_traffic.json"), "w") as f:
        json.dump(out, f, indent=2)
    plot(out)



if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(json.load(open(os.path.join(DATA_DIR, "early_warning_traffic.json"))))
    else:
        main()
