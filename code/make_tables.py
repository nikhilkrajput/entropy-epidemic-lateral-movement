"""
Generate the LaTeX tables of the article directly from the result files,
so that no table value is copied by hand.  Writes
Computer_Communications_Submission/tables/tab_*.tex, which main.tex \\inputs.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
_ms = os.path.join(ROOT, "Computer_Communications_Submission")
OUT = os.path.join(_ms, "tables") if os.path.isdir(_ms) else os.path.join(ROOT, "tables")
os.makedirs(OUT, exist_ok=True)


def J(*p):
    return json.load(open(os.path.join(ROOT, *p)))


def f2(x):
    return f"{x:.2f}"


def lead(x):
    return "never" if x == float("-inf") else f"${x:+.1f}$".replace("+-", "-")


def star(p):
    return "^{**}" if p < 0.001 else ("^{*}" if p < 0.05 else "")


ident = J("data", "identifiability_traffic.json")
ew = J("data", "early_warning_traffic.json")
ekf = J("data", "ekf_results.json")
hc = J("lanl_real_pipeline", "cyber1_full", "results", "hourly_checks.json")
bc = J("lanl_real_pipeline", "cyber1_full", "results", "cyber1_budget_curve.json")
ph = J("lanl_real_pipeline", "cyber1_full", "results", "cyber1_perhost_results.json")
gr = J("lanl_real_pipeline", "outputs", "run_58day_full", "growth.json")
nv = J("lanl_real_pipeline", "cyber1_full", "results", "cyber1_novelty.json")

# ---------------- Table: identifiability
rows = []
for sa, base in (("0.0", "none"), ("0.15", "iid")):
    for m in ("0.5", "1.0", "2.0", "5.0"):
        r = ident[f"m{m}_sa{sa}"]
        u = ident[f"m{m}_sa{sa}_offset"]
        snr = ew[f"m{m}_{base}"]["snr_on"]
        rows.append(f"{'0' if sa == '0.0' else '0.15'} & {float(m):g} & {f2(snr)} & {f2(r['beta'])} & "
                    f"{f2(r['gamma'])} & {f2(r['logR0'])} & {f2(r['logR0_R2_small_peak'])} & "
                    f"{f2(r['logR0_R2_large_peak'])} & {f2(r['auc_R0_gt_1'])} & {f2(r['balacc_R0_gt_1'])} & "
                    f"{f2(u['logR0'])} & {f2(u['logR0_R2_large_peak'])} & {f2(u['auc_R0_gt_1'])} \\\\")
o = ident["oracle_true_iota"]
ou = ident["oracle_true_iota_offset"]
n_small, n_large = o["n_small_peak"], o["n_large_peak"]
tab = r"""\begin{table}[t]
\caption{Random Forest on simulated logon traffic, 5-fold CV, 600 scenarios
(%d\%% with $R_0 > 1$). $m$: lateral logons per infected host per benign
logon of an average host; $s_a$: baseline variability. $R^2(\log R_0)$ is
given overall and for small (peak $\iota < 0.05$, %d scenarios) and large
(peak $\iota > 0.3$, %d scenarios) epidemics. AUC and BA: ROC AUC and
balanced accuracy of a classifier for $R_0 > 1$. Known seed time: every
epidemic starts at day 0; unknown: the seed is delayed by
$d \sim \mathcal{U}(0, 60)$ days.}
\label{tab:ident}
\centering
{\footnotesize\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}ccccccccccccc@{}}
\toprule
& & & \multicolumn{7}{c}{Known seed time} & \multicolumn{3}{c}{Unknown seed time} \\
\cmidrule(lr){4-10}\cmidrule(lr){11-13}
& & & & & \multicolumn{3}{c}{$R^2(\log R_0)$} & & & \multicolumn{2}{c}{$R^2(\log R_0)$} & \\
\cmidrule(lr){6-8}\cmidrule(lr){11-12}
$s_a$ & $m$ & SNR$_{\text{on}}$ & $R^2(\beta)$ & $R^2(\gamma)$ & all & small & large & AUC & BA & all & large & AUC \\
\midrule
%s
\midrule
\multicolumn{3}{@{}l}{Oracle (true $\iota$)} & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\
\bottomrule
\end{tabular}}
\end{table}
""" % (round(100 * ident["frac_R0_gt_1"]), n_small, n_large, "\n".join(rows),
       f2(o["beta"]), f2(o["gamma"]), f2(o["logR0"]), f2(o["logR0_R2_small_peak"]),
       f2(o["logR0_R2_large_peak"]), f2(o["auc_R0_gt_1"]), f2(o["balacc_R0_gt_1"]),
       f2(ou["logR0"]), f2(ou["logR0_R2_large_peak"]), f2(ou["auc_R0_gt_1"]))
open(os.path.join(OUT, "tab_ident.tex"), "w").write(tab)

# ---------------- Table: EKF
mc = ekf["misspecification"]
cs = ekf["correct_on_common_subset"]
tab = r"""\begin{table}[t]
\caption{EKF setting (per-scenario $H_0$ estimated from an attack-free
period; $\gamma$ supplied with 15\%% error). Upper block: the %d resolvable
scenarios (causal read-out: %d). Lower block: linkage misspecification,
on the %d scenarios resolvable under every assumption. RF: 5-fold CV.}
\label{tab:ekf}
\centering
\small
\begin{tabular}{@{}lccc@{}}
\toprule
Estimator & $R^2(\beta)$ & $R^2(R_0)$ & $R^2(\log R_0)$ \\
\midrule
Prior only, $0.325/\gamma_{\text{sup}}$ & -- & %s & %s \\
EKF, retrospective & %s & %s & %s \\
EKF, causal & %s & %s & %s \\
Oracle, $\beta/\gamma_{\text{sup}}$ & -- & %s & -- \\
Batch RF & %s & %s & %s \\
Batch RF, with $\gamma_{\text{sup}}$ & %s & %s & %s \\
\midrule
EKF, correct linkage & %s & %s & %s \\
\quad $\eta$ assumed $0.8\times$ & %s & %s & %s \\
\quad $\eta$ assumed $1.2\times$ & %s & %s & %s \\
\quad $H_0$ assumed $+0.1$ bits & %s & %s & %s \\
\quad $H_0$ assumed $-0.1$ bits & %s & %s & %s \\
\bottomrule
\end{tabular}
\end{table}
""" % (ekf["n_identifiable"], ekf["n_causal_available"], ekf["misspecification_common_subset_n"],
       f2(ekf["prior_only_subset"]["R0"]["R2"]), f2(ekf["prior_only_subset"]["R0"]["R2_logR0"]),
       f2(ekf["ekf_subset"]["beta"]["R2"]), f2(ekf["ekf_subset"]["R0"]["R2"]), f2(ekf["ekf_subset"]["R0"]["R2_logR0"]),
       f2(ekf["ekf_causal_subset"]["beta"]["R2"]), f2(ekf["ekf_causal_subset"]["R0"]["R2"]),
       f2(ekf["ekf_causal_subset"]["R0"]["R2_logR0"]),
       f2(ekf["oracle_beta_over_gamma_supplied_subset"]["R0"]["R2"]),
       f2(ekf["rf_subset"]["beta"]["R2"]), f2(ekf["rf_subset"]["R0"]["R2"]), f2(ekf["rf_subset"]["R0"]["R2_logR0"]),
       f2(ekf["rf_subset_with_gamma"]["beta"]["R2"]), f2(ekf["rf_subset_with_gamma"]["R0"]["R2"]),
       f2(ekf["rf_subset_with_gamma"]["R0"]["R2_logR0"]),
       f2(cs["beta"]["R2"]), f2(cs["R0"]["R2"]), f2(cs["R0"]["R2_logR0"]),
       *[f2(v) for k in ("eta_x0.8", "eta_x1.2", "H0_plus_0.1", "H0_minus_0.1")
         for v in (mc[k]["common_subset"]["beta"]["R2"], mc[k]["common_subset"]["R0"]["R2"],
                   mc[k]["common_subset"]["R0"]["R2_logR0"])])
open(os.path.join(OUT, "tab_ekf.tex"), "w").write(tab)

# ---------------- Table: early warning
rows = []
names = {"none": "none", "iid": "independent", "ar1": "AR(1)"}
for base in ("none", "iid", "ar1"):
    for m in ("0.5", "1.0", "2.0", "5.0"):
        v = ew[f"m{m}_{base}"]
        cells = []
        for k in ("LEVEL", "CUSUM", "RATE"):
            d = v[k]
            cells.append(f"{lead(d['lead_median_censored'])} & {d['missed']} & "
                         f"${d['early_fraction']:.2f}{star(d['early_vs_chance_p'])}$")
        rows.append(f"{names[base] if m == '0.5' else ''} & {float(m):g} & {f2(v['snr_on'])} & " + " & ".join(cells) + r" \\")
    if base != "ar1":
        rows.append(r"\addlinespace")
chance = [ew[k][d]["chance_early_fraction"] for k in ew if k.startswith("m") for d in ("LEVEL", "CUSUM", "RATE")]
far = [ew[k][d]["far_fresh"] for k in ew if k.startswith("thresholds") for d in ("LEVEL", "CUSUM", "RATE")]
n_onset = ew["m1.0_none"]["n_runs_with_onset"]
tab = r"""\begin{table}[t]
\caption{Early warning on simulated logon traffic (%d runs reaching onset).
Thresholds give 0.01 false alarms/day on 1{,}000 attack-free series
(re-measured on 1{,}000 fresh series: %.4f--%.4f). Lead: censored median,
days relative to onset (negative: after onset). Miss: runs never detected.
Early: fraction of runs alarming before onset; $^{*}p < 0.05$,
$^{**}p < 0.001$ against chance (%.2f--%.2f).}
\label{tab:ew}
\centering
{\small\setlength{\tabcolsep}{2.3pt}
\begin{tabular}{@{}lccccccccccc@{}}
\toprule
& & & \multicolumn{3}{c}{Level} & \multicolumn{3}{c}{CUSUM} & \multicolumn{3}{c}{Rate} \\
\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}
Baseline & $m$ & SNR$_{\text{on}}$ & lead & miss & early & lead & miss & early & lead & miss & early \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table}
""" % (n_onset, min(far), max(far), min(chance), max(chance), "\n".join(rows))
open(os.path.join(OUT, "tab_ew.tex"), "w").write(tab)

# ---------------- Table: LANL 2017 correlations
cr = gr["correlations"]
lab = {"H_S_dst": "$H^{dst}$ (destination host, Shannon)", "H_S_src": "$H^{src}$ (source host, Shannon)",
       "H_T_src": "Source host, Tsallis-2", "H_R_src": "Source host, R\\'enyi-2",
       "H_S_usr": "$H^{usr}$ (user name, Shannon)", "H_Sp": "Inter-arrival, spectral"}


rows = [f"{lab[k]} & ${cr[k]['r_window']:+.3f}$ \\\\"
        for k in ("H_S_dst", "H_S_src", "H_T_src", "H_R_src", "H_S_usr", "H_Sp")]
tab = r"""\begin{table}[t]
\centering
\caption{Pearson correlation of each entropy observable with the
per-window active-destination fraction $\iota_W$ (LANL 2017,
$n = 22{,}033$ windows). Windows overlap by half and are autocorrelated,
so no $p$-values are given.}
\label{tab:lanl_linkage}
\small
\begin{tabular}{@{}lc@{}}
\toprule
Observable & $r$ with $\iota_W$ \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows)
open(os.path.join(OUT, "tab_lanl_linkage.tex"), "w").write(tab)

# ---------------- Table: red-team hourly detectors (level-shift-free calibration)
rows = []
lab = {"H_dst": "Destination entropy", "k_dst": "Distinct destinations", "n": "Logon count"}
for col in ("H_dst", "k_dst", "n"):
    cells = []
    for k in ("LEVEL", "CUSUM", "RATE"):
        d = hc[col][k]
        cells.append(f"{d['alarms_in_rt_hours']}/{d['alarms_total']} (${d['p_rt_share_vs_hour_share']:.2g}$)")
    rows.append(f"{lab[col]} & " + " & ".join(cells) + r" \\")
tab = r"""\begin{table}[t]
\caption{LANL red-team data, network-wide hourly detectors. Each hour is
standardised against red-team-free hours of the same hour-of-day and
day-of-week cell, and thresholds give 0.1 false alarms per red-team-free
day. Entries: alarms in red-team hours / all alarms over the 58 days
(one-sided binomial $p$ against the red-team share of all hours, %.3f;
red-team hours are %.1f\%% of campaign hours).}
\label{tab:cyber1}
\centering
\small
\begin{tabular}{@{}lccc@{}}
\toprule
Observable & Level & CUSUM & Rate \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (hc["redteam_hours"] / hc["n_hours"], 100 * hc["redteam_share_of_campaign_hours"], "\n".join(rows))
open(os.path.join(OUT, "tab_cyber1.tex"), "w").write(tab)

# ---------------- Table: per-host scores
rows = []
lab = {"X_host": "Cross-entropy vs.\\ own history", "KL_host": "KL divergence vs.\\ own history",
       "new_dst": "New destinations", "k_dst": "Distinct destinations",
       "H_today": "Destination entropy", "n": "Logon count",
       "host_age": "Host age", "neg_log_hist": "$-\\log$ earlier logons"}


def nvc(sc, key):
    return f2(nv[sc][key]) if sc in nv else "--"


for sc in ("X_host", "KL_host", "new_dst", "k_dst", "H_today", "n", "host_age", "neg_log_hist"):
    b = bc[sc]
    ci = (f"{f2(nv[sc]['auc_all_source_ci95'][0])}--{f2(nv[sc]['auc_all_source_ci95'][1])}"
          if sc in nv else "--")
    rows.append(f"{lab[sc]} & {f2(b['auc_within_day_percentile'])} & {ci} & {nvc(sc, 'auc_with_history')} & "
                f"{nvc(sc, 'auc_within_age_stratum')} & {b['median_rank_of_positives']:.0f} & "
                f"{b['budget_for_25pct_recall']:.0f} & {b['recall_at_10']:.2f} \\\\")
    if sc == "n":
        rows.append(r"\multicolumn{8}{@{}l}{\emph{Novelty baselines}} \\")
tab = r"""\begin{table}[t]
\caption{LANL red-team data, per-host-day scores on campaign days (%d
red-team host-days from %d source hosts, among %s; median %s host-days per
day). AUC: within-day percentile ROC AUC. CI: 95\%% interval over source
hosts (cluster bootstrap). Hist.: AUC among host-days with at least %d days
of history (%d positives). Age: AUC within strata of host age. Rank: median
within-day rank of red-team host-days. $b_{25}$: daily alert budget
(top-$b$ host-days per day) that captures 25\%% of red-team host-days.
R@10: share captured at 10 alerts per day. --: not computed.}
\label{tab:cyber1host}
\centering
{\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lccccccc@{}}
\toprule
Score & AUC & CI & Hist. & Age & Rank & $b_{25}$ & R@10 \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table}
""" % (bc["positives"], len(nv["source_hosts"]), f"{bc['host_days']:,}".replace(",", "{,}"),
       f"{bc['median_host_days_per_day']:,.0f}".replace(",", "{,}"), nv["h_min_days"],
       nv["positives_with_history"], "\n".join(rows))
open(os.path.join(OUT, "tab_cyber1host.tex"), "w").write(tab)
print("tables written to", OUT)
