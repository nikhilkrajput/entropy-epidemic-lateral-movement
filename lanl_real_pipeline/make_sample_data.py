"""
Generate a TINY synthetic dataset in LANL wls-CSV format, purely so the
pipeline can be smoke-tested without access to the real (multi-GB) LANL data.

It writes sample_data/wls_day-0X_2v.csv for X = 1..7, with the canonical LANL
host-event columns and a realistic mix of:
  * EventID 4624 network (3) / remote-interactive (10) logons  -> kept
  * EventID 4634/4769 and self-loops and '-' sources           -> filtered out
The cumulative set of contacted destinations grows logistically across the
seven days, so the downstream logistic growth fit and entropy linkage are
non-degenerate.

THIS IS NOT REAL DATA. It exists only to prove the code runs end-to-end.
Point config.LANL_DATA_PATH at the genuine dataset for real results.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(HERE, "sample_data")
os.makedirs(SAMPLE_DIR, exist_ok=True)

SECONDS_PER_DAY = 86400
N_DAYS = 7
N_HOSTS = 160
EVENTS_PER_DAY = 4500

rng = np.random.default_rng(2017)
hosts = [f"C{i:04d}" for i in range(N_HOSTS)]
users = [f"U{i:03d}@DOM1" for i in range(40)] + [f"C{i:04d}$" for i in range(20)]

# Logistic growth of the reachable-destination frontier across days
# (saturates around day 4-5 so later days are not pure extrapolation tails).
day_idx = np.arange(1, N_DAYS + 1)
frontier = N_HOSTS / (1.0 + np.exp(-1.3 * (day_idx - 2.5)))
frontier = np.clip(frontier.astype(int), 12, N_HOSTS)

for d in day_idx:
    n_reach = int(frontier[d - 1])
    reachable = hosts[:n_reach]
    base = (d - 1) * SECONDS_PER_DAY
    times = np.sort(rng.integers(0, SECONDS_PER_DAY, size=EVENTS_PER_DAY)) + base

    rows = []
    for t in times:
        # Intra-day "campaign intensity" oscillates (~1.5 cycles/day) so the
        # per-window active-host diversity -- and hence iota -- varies within
        # the day. This same pattern repeats every day, which keeps the
        # entropy<->iota linkage learnable across days (so leave-one-day-out
        # CV behaves, instead of collapsing on a near-constant target).
        t_frac = (int(t) - base) / SECONDS_PER_DAY
        intensity = 0.5 * (1.0 + np.sin(2.0 * np.pi * 1.5 * t_frac))
        active_k = max(4, int(intensity * n_reach))
        active = reachable[:active_k]

        roll = rng.random()
        if roll < 0.15:
            # noise: a logoff / kerberos event that must be filtered out
            rows.append((int(t), int(rng.choice([4634, 4769])),
                         rng.choice(hosts), int(rng.choice([3, 10])),
                         rng.choice(hosts), rng.choice(users), "DOM1"))
            continue
        src = rng.choice(hosts)
        if roll < 0.25:
            dst = src                      # self-loop -> filtered out
        elif roll < 0.30:
            src = "-"                      # placeholder source -> filtered out
            dst = rng.choice(active)
        else:
            dst = rng.choice(active)
        rows.append((int(t), 4624, dst, int(rng.choice([3, 3, 3, 10])),
                     src, rng.choice(users), "DOM1"))

    df = pd.DataFrame(rows, columns=[
        "Time", "EventID", "LogHost", "LogonType", "Source", "UserName", "DomainName"])
    out = os.path.join(SAMPLE_DIR, f"wls_day-{d:02d}_2v.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df):,} rows)")

print("\nSample data ready. Now run:  python run_lanl_analysis.py")
