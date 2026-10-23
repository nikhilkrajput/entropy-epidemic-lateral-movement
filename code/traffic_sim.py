"""
Host-level authentication-traffic simulator used for the synthetic
experiments.  Entropies are computed from simulated logon events, so the
entropy--infection relation and the noise are produced by the traffic model
rather than assumed.

Network: N hosts.  Each observation window holds W logon events.
  * Benign logons: source host ~ Zipf(a_src), destination host ~ Zipf(a_dst)
    over independently permuted host ranks (popular servers receive most
    logons).  Optional window-to-window drift: a_dst varies per window as
    a_dst + s_a * z_w, z_w ~ N(0, 1).
  * Lateral movement: each infected host issues m logons for every benign
    logon issued by an average host, so in a window the expected share of
    malicious events is I*m / (N + I*m).  Malicious source = an infected
    host chosen uniformly; destination = a host chosen uniformly (fan-out
    to hosts the adversary has not yet visited is not distinguished).

The infected count I(t) follows the deterministic SIR solution; the
infected host identities are a fixed random subset of size I(t).

Default parameters are set so that benign windows resemble the LANL
2017 data used in the paper (median destination-host entropy about
2.5 bits, source-host entropy about 5.8 bits for W = 500).
"""
import numpy as np

N_HOSTS = 1000
W = 500
A_DST = 1.9
A_SRC = 1.15


def zipf_cdf(a, n=N_HOSTS):
    w = np.arange(1, n + 1, dtype=float) ** (-a)
    c = np.cumsum(w)
    return c / c[-1]


def _sample_rows(exponents, n_per_row, rng, n=N_HOSTS):
    """Sample n_per_row Zipf ranks per row; row r uses exponent exponents[r].
    Exponents are rounded to 0.001 and rows sharing a value share a CDF."""
    exps = np.round(np.asarray(exponents, float), 3)
    out = np.empty((len(exps), n_per_row), dtype=np.int64)
    u = rng.random((len(exps), n_per_row))
    for a in np.unique(exps):
        rows = exps == a
        out[rows] = np.searchsorted(zipf_cdf(a, n), u[rows], side="left")
    return np.clip(out, 0, n - 1)


def _entropy_rows(ids):
    """Shannon, Renyi-2 and Tsallis-2 entropies and support size per row."""
    n_rows, w = ids.shape
    s = np.sort(ids, axis=1)
    new = np.ones_like(s, dtype=bool)
    new[:, 1:] = s[:, 1:] != s[:, :-1]
    run = np.cumsum(new.ravel()) - 1                    # global run index
    cnt = np.bincount(run).astype(float)
    run_row = np.repeat(np.arange(n_rows), new.sum(axis=1))
    p = cnt / w
    hs = -np.bincount(run_row, weights=p * np.log2(p), minlength=n_rows)
    s2 = np.bincount(run_row, weights=p ** 2, minlength=n_rows)
    return hs, -np.log2(s2), 1.0 - s2, new.sum(axis=1)


class Network:
    def __init__(self, rng, a_dst=A_DST, a_src=A_SRC, n=N_HOSTS):
        self.n = n
        self.a_dst, self.a_src = a_dst, a_src
        self.dst_perm = rng.permutation(n)
        self.src_perm = rng.permutation(n)
        self.inf_order = rng.permutation(n)      # infected hosts = first I

    def windows(self, I_counts, rng, m=5.0, s_a=0.0, w=W, rho=0.0):
        """Entropy features for a sequence of windows with I_counts infected.

        Baseline variability: the destination exponent is a_dst + s_a * z_w,
        with z_w a stationary AR(1) process with lag-1 correlation rho
        (rho = 0: independent per window).  Benign traffic uses `rng` with a
        fixed number of draws; malicious traffic uses a separate stream
        derived from it, so the benign noise is identical for every m."""
        n_win = len(I_counts)
        I = np.asarray(I_counts, float)
        rng_mal = np.random.default_rng(rng.integers(2 ** 63))
        e = rng.standard_normal(n_win)
        z = np.empty(n_win)
        z[0] = e[0]
        for i in range(1, n_win):
            z[i] = rho * z[i - 1] + np.sqrt(1.0 - rho ** 2) * e[i]
        a = self.a_dst + s_a * z
        dst = self.dst_perm[_sample_rows(a, w, rng, self.n)]
        src = self.src_perm[_sample_rows(np.full(n_win, self.a_src), w, rng, self.n)]
        f_mal = I * m / (self.n + I * m)
        n_mal = rng_mal.binomial(w, f_mal)
        # overwrite the first n_mal events of each window with lateral logons
        col = np.arange(w)[None, :]
        mal = col < n_mal[:, None]
        if mal.any():
            Iint = np.maximum(np.rint(I).astype(int), 1)
            r_src = (rng_mal.random((n_win, w)) * Iint[:, None]).astype(int)
            mal_src = self.inf_order[r_src]
            mal_dst = rng_mal.integers(0, self.n, (n_win, w))
            dst = np.where(mal, mal_dst, dst)
            src = np.where(mal, mal_src, src)
        d_s, d_r, d_t, k = _entropy_rows(dst)
        s_s, _, _, _ = _entropy_rows(src)
        return {"H_dst": d_s, "R_dst": d_r, "T_dst": d_t, "H_src": s_s, "k_dst": k,
                "mal_share": n_mal / w}

