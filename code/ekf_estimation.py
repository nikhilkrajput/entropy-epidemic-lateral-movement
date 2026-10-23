"""
Extended Kalman Filter (EKF) for sequential SIR parameter tracking
from a synthetic traffic-entropy time series.

State:  xi_t = [beta_t, iota_t]^T  (gamma supplied externally)
Process model (integrated with N_SUB Euler sub-steps per interval):
    beta_{t+1}  = beta_t + w_beta,  w_beta  ~ N(0, sigma_beta^2)
    d iota/dt   = beta * iota * s - gamma * iota,  s = 1 - iota - r
    d r/dt      = gamma * iota   (propagated, gamma supplied)
Observation:
    H_t = H0 + eta * iota_t + v_t,  v_t ~ N(0, sigma_v^2)

The EKF is compared with a batch Random Forest on summary statistics of
the same trace, a prior-only baseline, and runs with a misspecified linkage.
All inputs are synthetic.  Each scenario has its own baseline H0, estimated
from an attack-free period before the filter starts.

Writes data/ekf_results.json and figures/fig1_ekf_tracking.(pdf|png).
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.metrics import r2_score


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'data')
FIG_DIR = os.path.join(HERE, '..', 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

N_SUB = 10                # Euler sub-steps per observation interval
BETA_PRIOR_MEAN = 0.325   # mean of the U(0.05, 0.60) scenario prior


def sir_ode(y, t, beta, gamma, N):
    S, I, R = y
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return [dS, dI, dR]


def simulate_sir(beta, gamma, N=1000, I0=1, T=150, n_steps=300):
    t = np.linspace(0.0, T, n_steps)
    sol = odeint(sir_ode, [N - I0, I0, 0], t, args=(beta, gamma, N))
    return t, sol[:, 1] / N


def synth_entropy_obs(iota, H0=3.5, eta=1.8, sigma_v=0.08, rng_local=None):
    if rng_local is None:
        rng_local = np.random.default_rng()
    return H0 + eta * iota + rng_local.normal(0.0, sigma_v, len(iota))


def run_ekf(H_obs, t, gamma_known, H0=3.5, eta=1.8, sigma_v=0.08,
            sigma_beta=1e-4, beta0=0.20, iota0=1e-3, n_sub=N_SUB):
    """EKF with state xi = [beta_t, iota_t]; gamma is supplied externally.

    A scalar entropy observation cannot identify beta and gamma jointly, so
    gamma is treated as known.  The SIR susceptible fraction is
    s = 1 - iota - rec, with rec propagated from the filtered iota.  The
    predict step integrates the SIR rate equation with n_sub Euler
    sub-steps and propagates the Jacobian through the same sub-steps.
    """
    n = len(H_obs)
    xi = np.array([beta0, iota0], dtype=float)
    P = np.diag([0.05, 1e-3]) ** 2
    Q = np.diag([sigma_beta ** 2, 0.0])
    R = sigma_v ** 2
    H_mat = np.array([[0.0, eta]])

    beta_hist = np.zeros(n)
    iota_hist = np.zeros(n)
    R0_hist = np.zeros(n)
    var_beta_hist = np.zeros(n)

    rec = 0.0
    for k in range(n):
        dt = float(t[k] - t[k - 1]) if k > 0 else (float(t[1] - t[0]) if n > 1 else 1.0)
        h = dt / n_sub

        # Predict with sub-stepping; F is the product of sub-step Jacobians.
        beta_p, iota_p = xi
        F = np.eye(2)
        rec_p = rec
        for _ in range(n_sub):
            s_p = max(1.0 - iota_p - rec_p, 0.0)
            Fs = np.eye(2)
            Fs[1, 0] = h * iota_p * s_p
            Fs[1, 1] = 1.0 + h * (beta_p * (s_p - iota_p) - gamma_known)
            F = Fs @ F
            iota_new = iota_p + h * (beta_p * iota_p * s_p - gamma_known * iota_p)
            rec_p = rec_p + h * gamma_known * iota_p
            iota_p = float(np.clip(iota_new, 1e-6, 0.999999))
        xi_pred = np.array([beta_p, iota_p])
        P_pred = F @ P @ F.T + Q

        # Update
        y_innov = float(H_obs[k] - (H0 + eta * iota_p))
        S_innov = float((H_mat @ P_pred @ H_mat.T)[0, 0] + R)
        K = (P_pred @ H_mat.T).flatten() / S_innov
        xi = xi_pred + K * y_innov
        P = (np.eye(2) - np.outer(K, H_mat.flatten())) @ P_pred

        xi[0] = float(np.clip(xi[0], 0.001, 1.0))
        xi[1] = float(np.clip(xi[1], 1e-6, 0.999999))
        rec = float(min(rec_p, 1.0 - xi[1]))

        beta_hist[k] = xi[0]
        iota_hist[k] = xi[1]
        R0_hist[k] = xi[0] / max(gamma_known, 1e-6)
        var_beta_hist[k] = max(P[0, 0], 0.0)

    return {'beta': beta_hist, 'iota': iota_hist, 'R0': R0_hist, 'var_beta': var_beta_hist}


CAUSAL_DELAY_DAYS = 10.0   # causal read-out: beta_hat this long after iota_hat > 0.02
N_PRE = 40                 # attack-free samples (20 days) used to estimate H0


def readout(out, t):
    """Retrospective and causal read-outs of beta.

    retrospective: mean beta_hat between first iota_hat > 0.02 and the peak
                   of iota_hat (needs the whole trace);
    causal:        beta_hat at the first time iota_hat > 0.02 plus
                   CAUSAL_DELAY_DAYS.
    Either is None when iota_hat never exceeds 0.02 (or the causal time is
    beyond the horizon, or the rise lasts fewer than five steps)."""
    ih = out['iota']
    if ih.max() < 0.02:
        return None, None
    g0, pk = int(np.argmax(ih > 0.02)), int(np.argmax(ih))
    retro = float(np.mean(out['beta'][g0:pk])) if pk - g0 >= 5 else None
    tc = t[g0] + CAUSAL_DELAY_DAYS
    causal = float(out['beta'][int(np.searchsorted(t, tc))]) if tc <= t[-1] else None
    return retro, causal


def trace_features(H):
    slope = float(np.polyfit(np.arange(len(H)), H, 1)[0])
    return np.array([H.mean(), H.max(), H.std(), slope,
                     np.percentile(H, 90), np.percentile(H, 10),
                     np.max(np.diff(H)), np.min(np.diff(H))])


def metrics(y_true, y_pred, R0_true=None, R0_pred=None):
    out = {'R2': float(r2_score(y_true, y_pred))}
    if R0_true is not None:
        out['R2_logR0'] = float(r2_score(np.log(R0_true), np.log(np.maximum(R0_pred, 1e-3))))
        out['acc_R0_gt_1'] = float(np.mean((R0_true > 1) == (R0_pred > 1)))
    return out


def eval_ekf_over_scenarios(n_scenarios=600, seed=42):
    """EKF versus batch Random Forest on the same synthetic entropy traces.

    * Each scenario has its own baseline H0 ~ U(3.5, 5.0).  The trace starts
      with N_PRE attack-free samples; the EKF uses their mean as H0_hat,
      eta and sigma_v are supplied, and gamma is supplied with 15% relative
      error (truncated to +-50%).
    * The RF (200 trees, depth 10, 5-fold CV) sees eight summary statistics
      of the attack-period trace minus H0_hat (the same information).
    * Prior-only baseline: R0 = E[beta] / gamma_supplied, E[beta] = 0.325.
    * Sign of R0 - 1 on all scenarios: unresolvable runs count as
      'sub-critical' for the EKF.
    * Misspecification: the filter assumes eta x0.8 / x1.2 or H0_hat +- 0.1.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_predict, KFold

    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_scenarios):
        beta = float(rng.uniform(0.05, 0.60))
        gamma = float(rng.uniform(0.02, 0.25))
        H0 = float(rng.uniform(3.5, 5.0))
        t, iota = simulate_sir(beta, gamma)
        H_pre = H0 + rng.normal(0.0, 0.08, N_PRE)
        H_obs = synth_entropy_obs(iota, H0=H0, sigma_v=0.08, rng_local=rng)
        g_sup = float(np.clip(gamma * (1.0 + rng.normal(0.0, 0.15)), gamma * 0.5, gamma * 1.5))
        rows.append((beta, gamma, t, H_obs, g_sup, float(H_pre.mean())))

    beta_all = np.array([r[0] for r in rows])
    R0_all = np.array([r[0] / r[1] for r in rows])
    g_sup_all = np.array([r[4] for r in rows])
    X_all = np.array([trace_features(r[3] - r[5]) for r in rows])

    def run_config(dH0=0.0, eta=1.8):
        return [readout(run_ekf(H_obs, t, gamma_known=g_sup, H0=H0h + dH0, eta=eta), t)
                for beta, gamma, t, H_obs, g_sup, H0h in rows]

    ro = run_config()
    ok = np.array([r[0] is not None for r in ro])
    okc = np.array([r[1] is not None for r in ro])
    bh = np.array([r[0] for r in ro if r[0] is not None])
    bc = np.array([r[1] for r in ro if r[1] is not None])
    R0_ekf = bh / g_sup_all[ok]

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rf_R0_sub = cross_val_predict(rf, X_all[ok], R0_all[ok], cv=kf)
    rf_beta_sub = cross_val_predict(rf, X_all[ok], beta_all[ok], cv=kf)
    rf_R0_all = cross_val_predict(rf, X_all, R0_all, cv=kf)
    rf_beta_all = cross_val_predict(rf, X_all, beta_all, cv=kf)
    prior_R0_sub = BETA_PRIOR_MEAN / g_sup_all[ok]
    prior_R0_all = BETA_PRIOR_MEAN / g_sup_all

    R0_ekf_all = np.full(n_scenarios, 0.5)
    R0_ekf_all[ok] = R0_ekf
    R0_causal_all = np.full(n_scenarios, 0.5)
    R0_causal_all[okc] = bc / g_sup_all[okc]
    sign = lambda pred: float(np.mean((R0_all > 1) == (pred > 1)))  # noqa: E731

    res = {
        'n_scenarios': n_scenarios,
        'frac_R0_gt_1': float(np.mean(R0_all > 1)),
        'n_identifiable': int(ok.sum()), 'n_unresolvable': int((~ok).sum()),
        'unresolvable_with_R0_gt_1': int(((~ok) & (R0_all > 1)).sum()),
        'n_causal_available': int(okc.sum()),
        'ekf_subset': {'beta': metrics(beta_all[ok], bh),
                       'R0': metrics(R0_all[ok], R0_ekf, R0_all[ok], R0_ekf)},
        'ekf_causal_subset': {'beta': metrics(beta_all[okc], bc),
                              'R0': metrics(R0_all[okc], bc / g_sup_all[okc],
                                            R0_all[okc], bc / g_sup_all[okc])},
        'rf_subset': {'beta': metrics(beta_all[ok], rf_beta_sub),
                      'R0': metrics(R0_all[ok], rf_R0_sub, R0_all[ok], rf_R0_sub)},
        'prior_only_subset': {'R0': metrics(R0_all[ok], prior_R0_sub, R0_all[ok], prior_R0_sub)},
        'oracle_beta_over_gamma_supplied_subset': {
            'R0': metrics(R0_all[ok], beta_all[ok] / g_sup_all[ok])},
        'rf_all': {'beta': metrics(beta_all, rf_beta_all),
                   'R0': metrics(R0_all, rf_R0_all, R0_all, rf_R0_all)},
        'sign_accuracy_all': {'majority': float(max(np.mean(R0_all > 1), np.mean(R0_all <= 1))),
                              'prior_only': sign(prior_R0_all),
                              'ekf_retrospective': sign(R0_ekf_all),
                              'ekf_causal': sign(R0_causal_all),
                              'rf': sign(rf_R0_all)},
        'R0_range_subset': [float(R0_all[ok].min()), float(R0_all[ok].max()),
                            float(np.median(R0_all[ok]))],
    }

    # RF also given the supplied gamma (like-for-like with the EKF)
    Xg = np.column_stack([X_all, g_sup_all])
    rf_R0_sub_g = cross_val_predict(rf, Xg[ok], R0_all[ok], cv=kf)
    rf_beta_sub_g = cross_val_predict(rf, Xg[ok], beta_all[ok], cv=kf)
    res['rf_subset_with_gamma'] = {'beta': metrics(beta_all[ok], rf_beta_sub_g),
                                   'R0': metrics(R0_all[ok], rf_R0_sub_g, R0_all[ok], rf_R0_sub_g)}
    # how the EKF's beta error relates to the error in the supplied gamma
    gamma_all = np.array([r[1] for r in rows])
    res['corr_beta_error_gamma_error'] = float(np.corrcoef(bh - beta_all[ok],
                                                           g_sup_all[ok] - gamma_all[ok])[0, 1])

    mis_runs = {'eta_x0.8': dict(eta=1.44), 'eta_x1.2': dict(eta=2.16),
                'H0_plus_0.1': dict(dH0=0.1), 'H0_minus_0.1': dict(dH0=-0.1)}
    mis_out = {name: run_config(**kw) for name, kw in mis_runs.items()}
    common = ok.copy()
    for rm in mis_out.values():
        common &= np.array([r[0] is not None for r in rm])
    base_c = np.array([ro[i][0] for i in np.flatnonzero(common)])
    res['misspecification_common_subset_n'] = int(common.sum())
    res['correct_on_common_subset'] = {
        'beta': metrics(beta_all[common], base_c),
        'R0': metrics(R0_all[common], base_c / g_sup_all[common], R0_all[common], base_c / g_sup_all[common])}
    mis = {}
    for name, rm in mis_out.items():
        okm = np.array([r[0] is not None for r in rm])
        bmh = np.array([r[0] for r in rm if r[0] is not None])
        bc_ = np.array([rm[i][0] for i in np.flatnonzero(common)])
        mis[name] = {'n_identifiable': int(okm.sum()),
                     'beta': metrics(beta_all[okm], bmh),
                     'R0': metrics(R0_all[okm], bmh / g_sup_all[okm],
                                   R0_all[okm], bmh / g_sup_all[okm]),
                     'common_subset': {'beta': metrics(beta_all[common], bc_),
                                       'R0': metrics(R0_all[common], bc_ / g_sup_all[common],
                                                     R0_all[common], bc_ / g_sup_all[common])}}
    res['misspecification'] = mis
    return res


def plot_demo_trace():
    """Plot one representative trace where the EKF tracks beta_t and R0_t."""
    rng = np.random.default_rng(11)
    beta_true, gamma_true = 0.35, 0.07
    t, iota = simulate_sir(beta=beta_true, gamma=gamma_true, T=150, n_steps=300)
    H_obs = synth_entropy_obs(iota, sigma_v=0.08, rng_local=rng)
    gamma_supplied = float(gamma_true * (1.0 + rng.normal(0.0, 0.15)))
    out = run_ekf(H_obs, t, gamma_known=gamma_supplied)
    R0_true = beta_true / gamma_true

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(t, iota, 'k-', lw=2.0, label='True $\\iota(t)$')
    axes[0].plot(t, out['iota'], 'r--', lw=1.5, label='EKF $\\hat{\\iota}(t)$')
    axes[0].set_xlabel('Time (days)')
    axes[0].set_ylabel('Infected fraction')
    axes[0].set_title('(a) Infected fraction')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, np.full_like(t, beta_true), 'k-', lw=1.6, label='True $\\beta$')
    axes[1].plot(t, out['beta'], 'b-', lw=1.4, label='EKF $\\hat{\\beta}_t$')
    axes[1].set_xlabel('Time (days)')
    axes[1].set_ylabel('$\\beta$ (per day)')
    axes[1].set_title('(b) Transmission rate')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    sd_R0 = np.sqrt(out['var_beta']) / max(gamma_supplied, 1e-6)
    axes[2].plot(t, np.full_like(t, R0_true), 'k-', lw=1.6, label='True $R_0$')
    axes[2].plot(t, out['R0'], 'r-', lw=1.4, label='$\\hat{\\beta}_t/\\gamma_{\\mathrm{sup}}$')
    axes[2].fill_between(t, out['R0'] - sd_R0, out['R0'] + sd_R0, color='r', alpha=0.2,
                         label='$\\pm 1\\sigma$ ($P_{\\beta\\beta}$ only)')
    axes[2].set_xlabel('Time (days)')
    axes[2].set_ylabel('$R_0$')
    axes[2].set_title(f'(c) Reproduction number, $\\gamma_{{\\mathrm{{sup}}}}$ = {gamma_supplied:.4f}')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig1_ekf_tracking.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIG_DIR, 'fig1_ekf_tracking.png'), bbox_inches='tight', dpi=300)
    plt.close()
    final = float(np.mean(out['beta'][-50:]))
    print(f"demo: gamma supplied={gamma_supplied:.4f} ({(gamma_supplied/gamma_true-1)*100:+.1f}%), "
          f"final beta_hat={final:.4f} ({(final/beta_true-1)*100:+.1f}%)")


if __name__ == '__main__':
    print("EKF parameter tracking - synthetic-data evaluation")
    plot_demo_trace()
    if '--plot-only' in sys.argv:
        sys.exit(0)
    print("\nEvaluating EKF on 600 random scenarios ...")
    res = eval_ekf_over_scenarios(n_scenarios=600, seed=42)
    print(json.dumps(res, indent=1))
    with open(os.path.join(DATA_DIR, 'ekf_results.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print("ekf_results.json saved.")
