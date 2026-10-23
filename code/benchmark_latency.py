"""
Per-window processing cost and model size.

Measures, for one W = 500-event window:
  * the four entropy measures on one categorical dimension,
  * one Random Forest prediction (200 trees, max_depth 10, 16 features),
  * one step of the EKF used in the article (ekf_estimation.run_ekf),
and the serialised size of the fitted Random Forest.

Writes data/benchmark_latency.json.
"""
import json
import os
import pickle
import platform
import time

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
W = 500
REPS = 2000


def entropies(ids, x):
    """Four entropies of one window: categorical ids and an inter-arrival series x."""
    _, counts = np.unique(ids, return_counts=True)
    p = counts / counts.sum()
    h_s = -np.sum(p * np.log2(p))
    h_r = -np.log2(np.sum(p ** 2))
    h_t = 1.0 - np.sum(p ** 2)
    ps = np.abs(np.fft.rfft(x - x.mean())) ** 2
    ps = ps[ps > 0] / ps.sum()
    h_sp = -np.sum(ps * np.log2(ps))
    return h_s, h_r, h_t, h_sp


def timeit(fn, reps=REPS):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3), float(np.percentile(ts, 95) * 1e3)


def main():
    rng = np.random.default_rng(0)
    # A forest of the article's size and input shape (200 trees, depth 10,
    # 600 scenarios x 16 features), fitted to a heavy-tailed target like R0.
    X = rng.standard_normal((600, 16))
    y = np.exp(X[:, :4].sum(axis=1) * 0.5 + rng.standard_normal(600) * 0.3)
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=1)
    rf.fit(X, y)
    x1 = X[:1]
    ids = rng.integers(0, 300, W)

    gaps = np.diff(np.sort(rng.random(W + 1)))
    ent_med, ent_p95 = timeit(lambda: entropies(ids, gaps))
    rf_med, rf_p95 = timeit(lambda: rf.predict(x1), reps=300)
    Xb = np.repeat(x1, 1000, axis=0)
    b_med, _ = timeit(lambda: rf.predict(Xb), reps=30)
    rf_batched = b_med / 1000.0
    # Time the filter actually used in the paper (ekf_estimation.run_ekf,
    # 10 Euler sub-steps per interval), per observation step.
    import sys
    sys.path.insert(0, HERE)
    from ekf_estimation import run_ekf, simulate_sir, synth_entropy_obs
    tt, io = simulate_sir(0.35, 0.07)
    Hs = synth_entropy_obs(io, rng_local=rng)
    tot_med, tot_p95 = timeit(lambda: run_ekf(Hs, tt, gamma_known=0.07), reps=50)
    ekf_med, ekf_p95 = tot_med / len(Hs), tot_p95 / len(Hs)
    size_mb = len(pickle.dumps(rf)) / 1e6

    out = {
        "window_events": W,
        "entropy_4_measures_ms": {"median": ent_med, "p95": ent_p95},
        "rf_predict_single_ms": {"median": rf_med, "p95": rf_p95},
        "rf_predict_per_window_batched_1000_ms": rf_batched,
        "ekf_step_ms": {"median": ekf_med, "p95": ekf_p95},
        "total_median_ms_single": ent_med + rf_med + ekf_med,
        "total_median_ms_batched": ent_med + rf_batched + ekf_med,
        "rf_pickle_MB": size_mb,
        "hardware": {"machine": platform.machine(), "processor": "Intel Core i7-1255U, 16 GB RAM (laptop)", "rf_n_jobs": 1,
                     "python": platform.python_version(), "numpy": np.__version__,
                     "sklearn": sklearn.__version__},
    }
    with open(os.path.join(DATA_DIR, "benchmark_latency.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
