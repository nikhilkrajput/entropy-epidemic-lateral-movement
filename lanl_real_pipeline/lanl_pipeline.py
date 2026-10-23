"""
Entropy-epidemic analysis on a unified LANL authentication-event table.

This reproduces, on real data, the methodology of the article's LANL section:
  1. cumulative-unique-destination trajectory  ->  iota(t)
  2. sliding-window (W flows, 50% overlap) multi-entropy features
  3. empirical linkage test  H_S^dst = H0 + eta * iota  + per-type Pearson r
  4. daily logistic fit of cumulative reach  ->  growth rate r, plateau K
  5. Random Forest  entropy -> iota  under random-5-fold and leave-one-day-out CV
  6. a 2-panel diagnostic figure + a JSON results dump + the per-window
     feature CSV

Everything is driven by the `cfg` module passed in; nothing is hard-coded to a
particular machine or dataset layout.
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, GroupKFold

ENTROPY_COLS = ["H_S_src", "H_S_dst", "H_S_usr", "H_R_src", "H_T_src", "H_Sp"]
SECONDS_PER_DAY = 86400


# --------------------------------------------------------------------------
# Entropy primitives (operate on category counts / a real-valued series)
# --------------------------------------------------------------------------
def _probs_from_counts(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return np.array([])
    p = counts / total
    return p[p > 0]


def shannon_entropy_counts(counts):
    p = _probs_from_counts(counts)
    return float(-np.sum(p * np.log2(p))) if p.size else 0.0


def renyi_entropy_counts(counts, alpha=2.0):
    p = _probs_from_counts(counts)
    if not p.size:
        return 0.0
    return float((1.0 / (1.0 - alpha)) * np.log2(np.sum(p ** alpha) + 1e-15))


def tsallis_entropy_counts(counts, q=2.0):
    p = _probs_from_counts(counts)
    if not p.size:
        return 0.0
    return float((1.0 / (q - 1.0)) * (1.0 - np.sum(p ** q)))


def spectral_entropy_series(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 4 or x.std() < 1e-12:
        return 0.0
    fft_vals = np.abs(np.fft.rfft(x - x.mean())) ** 2
    total = fft_vals.sum()
    if total < 1e-12:
        return 0.0
    p = fft_vals / total
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log2(p)))


def _safe_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan")
    r, p = pearsonr(x, y)
    return float(r), float(p)


def _logistic(t, K, r, t0):
    return K / (1.0 + np.exp(-r * (t - t0)))


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
def compute_window_features(events, cfg):
    """Slide a W-event window (stride S) over the chronological stream and
    compute the six entropy observables + the per-window active fraction."""
    src = events["src_host"].to_numpy()
    dst = events["dst_host"].to_numpy()
    usr = events["username"].to_numpy()
    t_sec = events["epoch_time"].to_numpy().astype(float)
    t0 = t_sec.min()
    t_rel = t_sec - t0
    day0 = int(t0 // SECONDS_PER_DAY)  # absolute day index of the first event

    all_hosts = pd.unique(np.concatenate([src, dst]))
    N = int(len(all_hosts))

    # cumulative unique destinations -> iota_cum
    seen = set()
    cum_unique = np.empty(len(dst), dtype=np.int64)
    for i, h in enumerate(dst):
        seen.add(h)
        cum_unique[i] = len(seen)

    W, S = cfg.WINDOW_FLOWS, cfg.WINDOW_STRIDE
    n_total = len(events)
    if n_total <= W:
        raise RuntimeError(
            f"Only {n_total} events after filtering, but WINDOW_FLOWS={W}. "
            f"Lower WINDOW_FLOWS in config.py or provide more data.")

    rows = []
    for start in range(0, n_total - W, S):
        end = start + W
        src_counts = pd.Series(src[start:end]).value_counts().to_numpy()
        dst_slice = dst[start:end]
        dst_counts = pd.Series(dst_slice).value_counts().to_numpy()
        usr_counts = pd.Series(usr[start:end]).value_counts().to_numpy()

        t_win = t_rel[start:end]
        inter = np.diff(t_win)
        inter = np.where(inter <= 0, 1e-3, inter)

        t_center = float(t_win[len(t_win) // 2])
        rows.append({
            "t_sec": t_center,
            "t_day": int((t_center + t0) // SECONDS_PER_DAY) - day0,
            "window_start": int(start),
            "iota": float(len(np.unique(dst_slice))) / N,
            "iota_cum": float(cum_unique[end - 1]) / N,
            "H_S_src": shannon_entropy_counts(src_counts),
            "H_S_dst": shannon_entropy_counts(dst_counts),
            "H_S_usr": shannon_entropy_counts(usr_counts),
            "H_R_src": renyi_entropy_counts(src_counts, alpha=2.0),
            "H_T_src": tsallis_entropy_counts(src_counts, q=2.0),
            "H_Sp": spectral_entropy_series(inter),
        })

    feat = pd.DataFrame(rows).reset_index(drop=True)
    return feat, N


# --------------------------------------------------------------------------
# Linkage, logistic fit, RF
# --------------------------------------------------------------------------
def calibrate_linkage(feat):
    iota = feat["iota"].to_numpy()
    H = feat["H_S_dst"].to_numpy()
    coeffs = np.polyfit(iota, H, 1)
    eta, H0 = float(coeffs[0]), float(coeffs[1])
    r, p = _safe_pearson(iota, H)
    resid = H - (H0 + eta * iota)
    per_type = {}
    for col in ENTROPY_COLS:
        rw, pw = _safe_pearson(feat["iota"].to_numpy(), feat[col].to_numpy())
        rc, pc = _safe_pearson(feat["iota_cum"].to_numpy(), feat[col].to_numpy())
        per_type[col] = {
            "pearson_r_vs_iota_window": rw, "pearson_p_vs_iota_window": pw,
            "pearson_r_vs_iota_cum": rc, "pearson_p_vs_iota_cum": pc,
        }
    return {
        "primary_observable": "H_S_dst",
        "H0_bits": H0, "eta_bits_per_iota": eta,
        "pearson_r": r, "pearson_p_value": p,
        "residual_sigma_bits": float(resid.std()),
        "all_entropy_types": per_type,
    }


def fit_daily_logistic(events, N, cfg):
    # Day index aligned to absolute 86400-second boundaries (matches the
    # convention epoch//86400), then shifted so the first day is 0. This is
    # robust to absolute-Unix-timestamp inputs yet reproduces LANL exactly
    # (whose epoch starts near 0/1).
    day0 = int(events["epoch_time"].min() // SECONDS_PER_DAY)
    day = (events["epoch_time"] // SECONDS_PER_DAY - day0).astype(int)
    days = sorted(day.unique())
    seen = set()
    daily_days, daily_I = [], []
    for d in days:
        for h in events.loc[day == d, "dst_host"].unique():
            seen.add(h)
        daily_days.append(int(d) + 1)
        daily_I.append(len(seen))
    daily_days = np.array(daily_days, dtype=float)
    daily_I = np.array(daily_I, dtype=float)
    daily_iota = daily_I / N

    fit = {"K_hosts": float(daily_I.max()), "r_per_day": float("nan"),
           "t0_day": float("nan")}
    if len(daily_days) >= 4:
        try:
            p0 = [float(daily_I.max() * 1.5), 0.5, float(daily_days.mean())]
            popt, _ = curve_fit(_logistic, daily_days, daily_I, p0=p0, maxfev=10000)
            K, r, t0d = (float(x) for x in popt)
            fit.update(K_hosts=K, r_per_day=r, t0_day=t0d)
        except Exception as e:  # noqa: BLE001
            print(f"  logistic fit failed ({e}); reporting raw saturation only")
    else:
        print(f"  only {len(daily_days)} day(s) of data; skipping logistic fit")

    return fit, daily_days, daily_iota


def random_forest_cv(feat, cfg):
    X = feat[ENTROPY_COLS].to_numpy()
    y = feat["iota"].to_numpy()
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1,
                               random_state=cfg.RANDOM_SEED)
    out = {"target": "per-window iota", "feature_cols": ENTROPY_COLS,
           "n_samples": int(len(X))}

    n_splits = min(5, len(X))
    if n_splits >= 2:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.RANDOM_SEED)
        s = cross_val_score(rf, X, y, cv=kf, scoring="r2")
        out["R2_random_kfold_mean"] = float(s.mean())
        out["R2_random_kfold_std"] = float(s.std())
        out["R2_random_kfold_folds"] = s.tolist()
        out["random_kfold_n_splits"] = int(n_splits)

    groups = feat["t_day"].to_numpy()
    n_days = len(np.unique(groups))
    if n_days >= 2:
        gkf = GroupKFold(n_splits=n_days)
        s = cross_val_score(rf, X, y, cv=gkf, groups=groups, scoring="r2")
        out["R2_leave_one_day_out_mean"] = float(s.mean())
        out["R2_leave_one_day_out_std"] = float(s.std())
        out["R2_leave_one_day_out_folds"] = s.tolist()
        out["leave_one_day_out_n_days"] = int(n_days)
    else:
        out["R2_leave_one_day_out_mean"] = None
        out["note"] = "Only one day present; leave-one-day-out CV unavailable."
    return out


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def make_figure(feat, link, fit, daily_days, daily_iota, N, out_pdf, out_png,
                suptitle="Entropy-Epidemic Framework on Real LANL Authentication Events"):
    iota = feat["iota"].to_numpy()
    H = feat["H_S_dst"].to_numpy()
    H0, eta, r = link["H0_bits"], link["eta_bits_per_iota"], link["pearson_r"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))

    ax = axes[0]
    ax.plot(daily_days, daily_iota, "ko-", lw=2.0, ms=8, label="Daily cumulative $\\iota$")
    if not np.isnan(fit["r_per_day"]):
        ts = np.linspace(daily_days.min() - 0.5, daily_days.max() + 0.5, 200)
        ax.plot(ts, _logistic(ts, fit["K_hosts"], fit["r_per_day"], fit["t0_day"]) / N,
                "r--", lw=1.6,
                label=f"Logistic: $r$={fit['r_per_day']:.2f}/d")
    ax.set_xlabel("Day"); ax.set_ylabel("$\\iota$ = unique dst / $N$")
    ax.set_title("(a) Real LANL cumulative-spread trajectory")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(iota, H, s=10, alpha=0.45, color="steelblue",
               label=f"Per-window (n={len(feat)})")
    xf = np.linspace(iota.min(), iota.max(), 50)
    ax.plot(xf, H0 + eta * xf, "r-", lw=1.8,
            label=f"OLS: $H={H0:.2f}+{eta:.1f}\\,\\iota$")
    ax.set_xlabel("Per-window active fraction $\\iota$")
    ax.set_ylabel("Shannon entropy of dst_host (bits)")
    ax.set_title(f"(b) Empirical linkage: Pearson $r={r:.3f}$")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


    plt.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close()


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run(events, cfg, data_path_label="<configured path>", is_sample=False):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("\nComputing per-window entropy features "
          f"(W={cfg.WINDOW_FLOWS}, stride={cfg.WINDOW_STRIDE}) ...")
    feat, N = compute_window_features(events, cfg)
    print(f"  {len(feat)} windows; network size N={N} hosts")
    feat_path = os.path.join(cfg.OUTPUT_DIR, "lanl_features.csv")
    feat.to_csv(feat_path, index=False)

    print("Calibrating entropy-epidemic linkage on real data ...")
    link = calibrate_linkage(feat)
    print(f"  H_S^dst ~ {link['H0_bits']:.3f} + {link['eta_bits_per_iota']:.1f} * iota "
          f"| Pearson r={link['pearson_r']:.4f}, p={link['pearson_p_value']:.2e}")

    print("Fitting daily logistic growth of cumulative reach ...")
    fit, daily_days, daily_iota = fit_daily_logistic(events, N, cfg)
    if not np.isnan(fit["r_per_day"]):
        print(f"  r={fit['r_per_day']:.3f}/d, K={fit['K_hosts']:.0f} hosts")

    print("Random Forest: entropy -> iota ...")
    rf = random_forest_cv(feat, cfg)
    if "R2_random_kfold_mean" in rf:
        print(f"  random {rf.get('random_kfold_n_splits')}-fold CV R2={rf['R2_random_kfold_mean']:.4f}")
    if rf.get("R2_leave_one_day_out_mean") is not None:
        print(f"  leave-one-day-out CV R2={rf['R2_leave_one_day_out_mean']:.4f} "
              f"({rf['leave_one_day_out_n_days']} days)")

    out_pdf = os.path.join(cfg.OUTPUT_DIR, cfg.FIGURE_BASENAME + ".pdf")
    out_png = os.path.join(cfg.OUTPUT_DIR, cfg.FIGURE_BASENAME + ".png")
    suptitle = ("[SYNTHETIC SAMPLE - NOT REAL DATA] Entropy-Epidemic Smoke Test"
                if is_sample
                else "Entropy-Epidemic Framework on Real LANL Authentication Events")
    make_figure(feat, link, fit, daily_days, daily_iota, N, out_pdf, out_png,
                suptitle=suptitle)
    print(f"Figure written: {out_pdf}")

    if cfg.PAPER_FIGURES_DIR and is_sample:
        print("Skipping copy to paper figures/ (synthetic sample -> would "
              "overwrite a real-data figure). Set a real --data-path to enable it.")
    elif cfg.PAPER_FIGURES_DIR:
        os.makedirs(cfg.PAPER_FIGURES_DIR, exist_ok=True)
        import shutil
        paper_base = getattr(cfg, "PAPER_FIGURE_BASENAME", cfg.FIGURE_BASENAME)
        for ext in (".pdf", ".png"):
            dst = os.path.join(cfg.PAPER_FIGURES_DIR, paper_base + ext)
            shutil.copy(os.path.join(cfg.OUTPUT_DIR, cfg.FIGURE_BASENAME + ext), dst)
        print(f"Figure copied into paper: "
              f"{os.path.join(cfg.PAPER_FIGURES_DIR, paper_base + '.pdf')}")

    results = {
        "source_path": data_path_label,
        "n_events_total": int(len(events)),
        "n_hosts_network": N,
        "window_n_flows": cfg.WINDOW_FLOWS,
        "window_stride": cfg.WINDOW_STRIDE,
        "n_windows": int(len(feat)),
        "linkage_calibrated_on_real_data": link,
        "logistic_growth_fit": fit,
        "daily_aggregate": {"day": daily_days.tolist(),
                            "cum_iota": daily_iota.tolist()},
        "real_data_random_forest": rf,
    }
    res_path = os.path.join(cfg.OUTPUT_DIR, "lanl_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written: {res_path}")
    return results
