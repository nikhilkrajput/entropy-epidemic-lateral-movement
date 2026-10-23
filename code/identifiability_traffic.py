"""
Identifiability of SIR parameters from entropies of simulated logon traffic
(traffic_sim.py).  Replaces the linear entropy-like generator.

600 scenarios, identical across conditions: beta ~ U(0.05, 0.60),
gamma ~ U(0.02, 0.25), N = 1000, I0 = 1, 150 days, 20 windows of W = 500
logons at days 10..140.  Conditions: lateral intensity m (logons per
infected host per average benign logon) and baseline drift s_a.

Per window: destination-host Shannon, Renyi-2 and Tsallis-2 entropies and
source-host Shannon entropy.  Features: mean, max, s.d. and slope over the
20 windows for each of the four observables (16 features).

Estimators (5-fold CV):
  * RF regressor (200 trees, depth 10) for beta, gamma, R0, log R0, peak;
  * RF classifier for R0 > 1: ROC AUC and balanced accuracy;
  * oracle: the same models on the TRUE infected fraction at the 20 windows
    (upper bound on what any observable of iota at these windows can give);
  * ablation: one 4-feature block per observable.

Seed time: in the main runs the seed is at day 0 for every scenario.  The
`_offset` runs shift each scenario by an unknown delay d ~ U(0, 60) days
(the windows stay at days 10..140), so that features cannot rely on a
known seed time.

Writes data/identifiability_traffic.json.
"""
import json
import os
import sys

import numpy as np
from scipy.integrate import odeint
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import traffic_sim as TS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
N, N_SC, N_W = TS.N_HOSTS, 600, 20
T = np.linspace(0, 150, 1501)
W_IDX = [int(np.argmin(np.abs(T - w))) for w in np.linspace(10, 140, N_W)]
CONDITIONS = [(m, s) for s in (0.0, 0.15) for m in (0.5, 1.0, 2.0, 5.0)]
OBS = ["H_dst", "R_dst", "T_dst", "H_src"]


def sir(y, _t, b, g):
    S, I, R = y
    return [-b * S * I / N, b * S * I / N - g * I, g * I]


def block(x):
    return [x.mean(), x.max(), x.std(), np.polyfit(np.arange(len(x)), x, 1)[0]]


def evaluate(X, beta, gamma, R0, peak):
    reg = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
    kf = KFold(5, shuffle=True, random_state=0)
    res = {}
    for name, y in [("beta", beta), ("gamma", gamma), ("R0", R0),
                    ("logR0", np.log(R0)), ("peak", peak)]:
        pred = cross_val_predict(reg, X, y, cv=kf)
        res[name] = float(r2_score(y, pred))
        if name in ("logR0", "beta"):
            # accuracy by epidemic size (peak infected fraction)
            for lo, hi, tag in ((0, 0.05, "small"), (0.05, 0.3, "medium"), (0.3, 1.01, "large")):
                sel = (peak >= lo) & (peak < hi)
                res[f"{name}_R2_{tag}_peak"] = float(r2_score(y[sel], pred[sel]))
                res[f"n_{tag}_peak"] = int(sel.sum())
    lab = (R0 > 1).astype(int)
    prob = cross_val_predict(clf, X, lab, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                             method="predict_proba")[:, 1]
    res["auc_R0_gt_1"] = float(roc_auc_score(lab, prob))
    res["balacc_R0_gt_1"] = float(balanced_accuracy_score(lab, prob > 0.5))
    return res


def main():
    rng = np.random.default_rng(42)
    sc = []
    for i in range(N_SC):
        b, g = rng.uniform(0.05, 0.60), rng.uniform(0.02, 0.25)
        iota = odeint(sir, [N - 1, 1, 0], T, args=(b, g))[:, 1] / N
        sc.append((b, g, iota[W_IDX], iota.max(), iota))
    beta = np.array([s[0] for s in sc]); gamma = np.array([s[1] for s in sc])
    R0 = beta / gamma; peak = np.array([s[3] for s in sc])
    out = {"n_scenarios": N_SC, "frac_R0_gt_1": float(np.mean(R0 > 1)),
           "R0_range": [float(R0.min()), float(R0.max())]}

    X_or = np.array([list(s[2]) + block(s[2]) for s in sc])
    out["oracle_true_iota"] = evaluate(X_or, beta, gamma, R0, peak)
    print("oracle", {k: round(v, 3) for k, v in out["oracle_true_iota"].items()})

    # unknown seed time: shift each scenario by d ~ U(0, 60) days
    d_off = np.random.default_rng(4242).uniform(0, 60, N_SC)
    iw_off = [np.interp(T[W_IDX] - d, T, s[4], left=0.0) for s, d in zip(sc, d_off)]
    X_or = np.array([list(w) + block(w) for w in iw_off])
    out["oracle_true_iota_offset"] = evaluate(X_or, beta, gamma, R0, peak)
    print("oracle offset", {k: round(v, 3) for k, v in out["oracle_true_iota_offset"].items()})
    out["seed_offset_days"] = [0, 60]

    for m, s_a in CONDITIONS:
        feats, blocks = [], {o: [] for o in OBS}
        for i, (b, g, iw, pk, _) in enumerate(sc):
            r = np.random.default_rng(10_000 + i)          # paired across conditions
            net = TS.Network(r)
            o = net.windows(np.rint(iw * N), r, m=m, s_a=s_a)
            row = []
            for ob in OBS:
                bl = block(o[ob]); blocks[ob].append(bl); row += bl
            feats.append(row)
        key = f"m{m}_sa{s_a}"
        res = evaluate(np.array(feats), beta, gamma, R0, peak)
        res["ablation_R2_R0"] = {ob: float(r2_score(R0, cross_val_predict(
            RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42),
            np.array(blocks[ob]), R0, cv=KFold(5, shuffle=True, random_state=0)))) for ob in OBS}
        # empirical onset SNR: entropy shift at iota = 0.01 over baseline s.d.
        r = np.random.default_rng(7)
        net = TS.Network(r)
        base = net.windows(np.zeros(2000), r, m=m, s_a=s_a)["H_dst"]
        on = net.windows(np.full(2000, 10), r, m=m, s_a=s_a)["H_dst"]
        res["snr_on"] = float((on.mean() - base.mean()) / base.std())
        out[key] = res
        print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})
        feats = []
        for i, iw in enumerate(iw_off):
            r = np.random.default_rng(10_000 + i)
            o = TS.Network(r).windows(np.rint(iw * N), r, m=m, s_a=s_a)
            feats.append([v for ob in OBS for v in block(o[ob])])
        out[key + "_offset"] = evaluate(np.array(feats), beta, gamma, R0, peak)
        print(key + "_offset", {k: round(v, 3) for k, v in out[key + "_offset"].items()})
    with open(os.path.join(DATA_DIR, "identifiability_traffic.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
