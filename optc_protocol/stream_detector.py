"""
Streaming per-host detector: one pass over a time-ordered stream of
(time, src, dst) logons or flows, constant work per event, state of one
counter per (source, destination) pair seen.

Per event it updates today's counters of the source host and the event's
contribution to two scores, both computed against the host's history
before today (yesterday's state, frozen during the day):
  * cross-entropy X(s) = mean of -log2 (c_<t(s,d) + 1) / (c_<t(s) + K);
  * EB novelty     E(s) = -ln P(Binomial(n_t(s), q_s) >= k_t(s)), with k_t(s)
    today's first contacts and q_s = (D_<t(s) + a q_pop) / (N_<t(s) + a),
    q_pop the network-wide first-contact rate of earlier days.
At the end of each day (or whenever queried) the day's host scores are
available, and two alert rules are provided:
  * top-b: the b highest-scoring hosts of the day (the batch rule used in
    the evaluation);
  * threshold: alert as soon as a host's running score exceeds tau, the
    median over the previous W days of the day's b-th largest final score
    (a SIEM-style rule that fires during the day).

Usage:
  python stream_detector.py events.csv [--a A] [--K K] [--budget B]
      events.csv: time,src,dst (integer time in seconds, any host ids),
      time-ordered.  Writes per-day top-b alerts to stdout and a timing and
      memory summary to stderr.
"""
import argparse
import csv
import math
import sys
import time
from collections import defaultdict

import numpy as np
from scipy.stats import binom

DAY = 86400


class PerHostDetector:
    def __init__(self, K, a, window=7):
        self.K, self.a, self.window = K, a, window
        self.hist_sd = defaultdict(int)   # (s, d) -> logons before today
        self.hist_n = defaultdict(int)    # s -> logons before today
        self.hist_dst = defaultdict(int)  # s -> distinct destinations before today
        self.pop_new = 0                  # first contacts before today, network-wide
        self.pop_n = 0
        self.day = None
        self._reset_today()
        self.kth = []                     # b-th largest final E per past day (threshold rule)

    def _reset_today(self):
        self.n = defaultdict(int)         # s -> logons today
        self.k = defaultdict(int)         # s -> first contacts today
        self.surp = defaultdict(float)    # s -> summed cross-entropy surprisal today (bits)
        self.today_sd = defaultdict(int)  # (s, d) -> logons today

    # ---- per event: O(1) dictionary work, no logarithm of a sum
    def update(self, t, s, d):
        day = t // DAY
        if self.day is None:
            self.day = day
        elif day != self.day:
            self.close_day()
            self.day = day
        key = (s, d)
        c_hist = self.hist_sd.get(key, 0)
        self.surp[s] += -math.log2((c_hist + 1.0) / (self.hist_n.get(s, 0) + self.K))
        if c_hist == 0 and key not in self.today_sd:
            self.k[s] += 1
        self.today_sd[key] += 1
        self.n[s] += 1

    def q_pop(self):
        return (self.pop_new + 1.0) / (self.pop_n + 2.0)

    def eb_score(self, s):
        q = (self.hist_dst.get(s, 0) + self.a * self.q_pop()) / (self.hist_n.get(s, 0) + self.a)
        return float(-binom.logsf(self.k[s] - 1, self.n[s], q))

    def scores(self):
        """Today's scores so far: {s: (n, k, X, E)}; one vectorised binomial tail for all hosts."""
        hosts = list(self.n)
        n = np.fromiter((self.n[s] for s in hosts), float, len(hosts))
        k = np.fromiter((self.k[s] for s in hosts), float, len(hosts))
        hn = np.fromiter((self.hist_n.get(s, 0) for s in hosts), float, len(hosts))
        hdst = np.fromiter((self.hist_dst.get(s, 0) for s in hosts), float, len(hosts))
        q = (hdst + self.a * self.q_pop()) / (hn + self.a)
        e = -binom.logsf(k - 1, n, q)
        return {s: (self.n[s], self.k[s], self.surp[s] / self.n[s], float(e[i])) for i, s in enumerate(hosts)}

    def threshold(self):
        if not self.kth:
            return math.inf
        w = sorted(self.kth[-self.window:])
        return w[len(w) // 2]

    def close_day(self, budget=None):
        sc = self.scores()
        if budget:
            e = sorted((v[3] for v in sc.values()), reverse=True)
            self.kth.append(e[budget - 1] if len(e) >= budget else 0.0)
        for (s, d), c in self.today_sd.items():
            self.hist_sd[(s, d)] += c
        for s in self.n:
            self.hist_n[s] += self.n[s]
            self.hist_dst[s] += self.k[s]
            self.pop_new += self.k[s]
            self.pop_n += self.n[s]
        self.last_scores = sc
        self._reset_today()
        return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("--a", type=float, default=1e5)
    ap.add_argument("--K", type=int, default=17684)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--types", default="", help="comma-separated values of a 4th column to keep (e.g. LANL logon types)")
    args = ap.parse_args()
    keep = set(args.types.split(",")) if args.types else None
    det = PerHostDetector(args.K, args.a)
    n_ev, t_upd, t_close = 0, 0.0, 0.0
    w = csv.writer(sys.stdout)
    w.writerow(["day", "rank", "src", "n", "first_contacts", "X_bits", "E_nats"])

    def emit(day, sc):
        top = sorted(sc.items(), key=lambda kv: -kv[1][3])[:args.budget]
        for r, (s, v) in enumerate(top, 1):
            w.writerow([day, r, s, v[0], v[1], f"{v[2]:.4f}", f"{v[3]:.4f}"])

    with open(args.events, newline="") as f:
        for row in csv.reader(f):
            if keep is not None and row[3] not in keep:
                continue
            t, s, d = int(row[0]), row[1], row[2]
            day = t // DAY
            if det.day is not None and day != det.day:
                t0 = time.perf_counter()
                d0 = det.day
                sc = det.close_day(args.budget)
                t_close += time.perf_counter() - t0
                emit(d0, sc)
                det.day = day
            t0 = time.perf_counter()
            det.update(t, s, d)
            t_upd += time.perf_counter() - t0
            n_ev += 1
            if args.max_events and n_ev >= args.max_events:
                break
    d0 = det.day
    emit(d0, det.close_day(args.budget))
    print(f"events {n_ev} update_us_per_event {1e6 * t_upd / n_ev:.3f} close_s_total {t_close:.2f} "
          f"days {len(det.kth)} pairs {len(det.hist_sd)} hosts {len(det.hist_n)}", file=sys.stderr)


if __name__ == "__main__":
    main()
