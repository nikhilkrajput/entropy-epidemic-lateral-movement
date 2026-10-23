"""
Regressor comparison on the LANL 2017 windows (outputs/run_58day_full/
lanl_features.csv; no raw data needed).

Adds temporal-context features (entropy rate, rolling mean/s.d./slope,
cross-entropy ratios) to the six entropy features and compares seven
regressors (mean baseline, linear, ridge, kNN, Random Forest, Extra Trees,
histogram gradient boosting) for predicting the per-window active fraction
under (a) random 5-fold CV, which is optimistic because sliding windows
overlap, and (b) day-blocked group CV, in which no day is split between
training and test folds.

Writes model_comparison.json and fig_model_comparison.(pdf|png) (the figure
is not used in the article).
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              HistGradientBoostingRegressor)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_COLS = ["H_S_src", "H_S_dst", "H_S_usr", "H_R_src", "H_T_src", "H_Sp"]
SEED = 42
DAYBLOCK_FOLDS = 10   # day-grouped folds (each holds out ~1/10 of the days)


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Feature engineering: temporal context on top of the 6 baseline entropies
# ---------------------------------------------------------------------------
def build_enhanced_features(feat):
    df = feat.sort_values("window_start").reset_index(drop=True).copy()
    out = df[BASE_COLS].copy()
    out["dst_src_ratio"] = df["H_S_dst"] / (df["H_S_src"] + 1e-9)
    out["dst_usr_gap"] = df["H_S_dst"] - df["H_S_usr"]
    out["renyi_shannon_gap"] = df["H_S_src"] - df["H_R_src"]
    W = 5
    for col in ["H_S_dst", "H_S_src", "H_S_usr"]:
        s = df[col]
        out[f"{col}_rate"] = s.diff().fillna(0.0)
        out[f"{col}_rmean"] = s.rolling(W, min_periods=1).mean()
        out[f"{col}_rstd"] = s.rolling(W, min_periods=1).std().fillna(0.0)
    out["H_S_dst_slope"] = df["H_S_dst"].rolling(W, min_periods=2).apply(
        lambda x: float(np.polyfit(np.arange(len(x)), x, 1)[0]), raw=True).fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Model panel (tree models use n_jobs=1 so CV folds parallelise cleanly)
# ---------------------------------------------------------------------------
def model_panel():
    return {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "kNN (k=15)": make_pipeline(StandardScaler(),
                                    KNeighborsRegressor(n_neighbors=15)),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=10,
                                              n_jobs=1, random_state=SEED),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=200, n_jobs=1,
                                          random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=SEED),
    }


def evaluate(X, y, groups):
    rkf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    n_splits = min(DAYBLOCK_FOLDS, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for name, model in model_panel().items():
        rr = cross_val_score(model, X, y, cv=rkf, scoring="r2", n_jobs=-1)
        rg = cross_val_score(model, X, y, cv=gkf, groups=groups, scoring="r2",
                             n_jobs=-1)
        rows.append({"model": name,
                     "random_r2_mean": float(rr.mean()), "random_r2_std": float(rr.std()),
                     "dayblock_r2_mean": float(rg.mean()), "dayblock_r2_std": float(rg.std())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(HERE, "outputs", "run_58day_full", "lanl_features.csv"))
    ap.add_argument("--results", default=os.path.join(HERE, "outputs", "run_58day_full", "lanl_results.json"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "outputs", "run_58day_full"))
    args = ap.parse_args()

    feat = pd.read_csv(args.features)
    with open(args.results) as f:
        res = json.load(f)
    N = res["n_hosts_network"]
    y = feat["iota"].to_numpy()
    groups = feat["t_day"].to_numpy()
    log(f"Loaded {len(feat)} windows, {len(np.unique(groups))} days, N={N}")

    X_base = feat[BASE_COLS].to_numpy()
    Xe = build_enhanced_features(feat)
    enh_cols = list(Xe.columns)
    X_enh = Xe.to_numpy()
    log(f"Baseline feats={len(BASE_COLS)}  enhanced feats={len(enh_cols)}")

    log("Evaluating baseline feature set ...")
    base_rows = evaluate(X_base, y, groups)
    log("Evaluating enhanced feature set ...")
    enh_rows = evaluate(X_enh, y, groups)
    enh_by = {r["model"]: r for r in enh_rows}

    log(f"\n{'model':22s} {'BASE rand':>10s} {'BASE blk':>9s} {'ENH rand':>10s} {'ENH blk':>9s}")
    table = []
    for b in base_rows:
        e = enh_by[b["model"]]
        table.append({"model": b["model"],
                      "base_random_r2": b["random_r2_mean"], "base_dayblock_r2": b["dayblock_r2_mean"],
                      "enh_random_r2": e["random_r2_mean"], "enh_dayblock_r2": e["dayblock_r2_mean"],
                      "enh_dayblock_std": e["dayblock_r2_std"]})
        log(f"{b['model']:22s} {b['random_r2_mean']:10.4f} {b['dayblock_r2_mean']:9.4f} "
            f"{e['random_r2_mean']:10.4f} {e['dayblock_r2_mean']:9.4f}")

    best = max(table, key=lambda r: r["enh_dayblock_r2"])
    rf_base = next(r for r in table if r["model"] == "RandomForest")
    log(f"\nPaper baseline (RandomForest + 6 entropy feats): "
        f"blocked R2={rf_base['base_dayblock_r2']:.4f}, random R2={rf_base['base_random_r2']:.4f}")
    log(f"BEST ({best['model']} + enhanced feats): "
        f"blocked R2={best['enh_dayblock_r2']:.4f}, random R2={best['enh_random_r2']:.4f}")
    log(f"Improvement (blocked R2): {rf_base['base_dayblock_r2']:.4f} -> {best['enh_dayblock_r2']:.4f} "
        f"(+{best['enh_dayblock_r2'] - rf_base['base_dayblock_r2']:.4f})")


    summary = {
        "n_windows": int(len(feat)), "n_days": int(len(np.unique(groups))), "n_hosts": N,
        "dayblock_folds": int(min(DAYBLOCK_FOLDS, len(np.unique(groups)))),
        "linkage_r_HSdst": res["linkage_calibrated_on_real_data"]["pearson_r"],
        "enhanced_feature_names": enh_cols,
        "model_comparison": table,
        "baseline_paper_model": {"model": "RandomForest", "features": "6 entropy",
                                 "dayblock_r2": rf_base["base_dayblock_r2"],
                                 "random_r2": rf_base["base_random_r2"]},
        "best_model": {"model": best["model"], "features": "enhanced",
                       "dayblock_r2": best["enh_dayblock_r2"],
                       "dayblock_std": best["enh_dayblock_std"],
                       "random_r2": best["enh_random_r2"]},
    }
    out_json = os.path.join(args.outdir, "model_comparison.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSaved {out_json}")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    axes = [ax]
    names = [r["model"] for r in table]
    xb = np.arange(len(names))
    axes[0].barh(xb - 0.2, [r["base_dayblock_r2"] for r in table], height=0.38,
                 label="6 entropy feats", color="#4472C4")
    axes[0].barh(xb + 0.2, [r["enh_dayblock_r2"] for r in table], height=0.38,
                 label="enhanced feats", color="#70AD47")
    axes[0].set_yticks(xb); axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_xlabel("Day-blocked CV $R^2$ (per-window $\\iota$)")
    axes[0].set_title("(a) Model comparison on real 58-day LANL")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, axis="x"); axes[0].axvline(0, color="k", lw=0.8)

    plt.tight_layout()
    for ext in (".pdf", ".png"):
        plt.savefig(os.path.join(args.outdir, "fig_model_comparison" + ext), bbox_inches="tight", dpi=300)
    plt.close()
    log(f"Saved fig_model_comparison.pdf/.png")


if __name__ == "__main__":
    main()
