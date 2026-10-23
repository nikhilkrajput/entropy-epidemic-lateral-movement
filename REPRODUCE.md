# Reproducing the article's results

Each item lists the script that produces it and the output file it reads
from or writes to. The article's tables are generated from these outputs by
`code/make_tables.py`, and `code/check_claims.py` recomputes every number
quoted in the text.

| Article item | Script | Output |
|---|---|---|
| Table 1 (identifiability, known and unknown seed time) | `code/identifiability_traffic.py` | `data/identifiability_traffic.json` (SNR column from `data/early_warning_traffic.json`) |
| Table 2, Fig. 1 (EKF) | `code/ekf_estimation.py` (`--plot-only` redraws Fig. 1) | `data/ekf_results.json`, `figures/fig1_ekf_tracking.*` |
| Table 3, Fig. 2 (early warning) | `code/early_warning_traffic.py` (`--plot-only` redraws Fig. 2) | `data/early_warning_traffic.json`, `figures/fig2_early_warning_snr.*` |
| Proposition 1 thresholds, Corollary 1 delay, simulator check | `code/detectability_bound.py` | `data/detectability_bound.json` |
| Table 4, Fig. 3 (LANL 2017) | `lanl_real_pipeline/regen_figs_final.py` | `lanl_real_pipeline/outputs/run_58day_full/growth.json`, `figures/fig3_lanl_reach_linkage.*` |
| Section 7 checks (support size, next-window prediction, coverage) | `lanl_real_pipeline/lanl_extra_checks.py` | `.../run_58day_full/extra_checks.json` |
| LANL 2017 day-blocked regressor comparison (Section 7) | `lanl_real_pipeline/enhanced_analysis.py` | `.../run_58day_full/model_comparison.json` |
| LANL 2017 leave-one-day-out CV (Section 7) | `lanl_real_pipeline/run_lanl_analysis.py` | `.../run_58day_full/lanl_results.json` |
| LANL 2017 window features (needs raw data) | `lanl_real_pipeline/run_lanl_analysis.py --max-bytes-per-file 50331648 --output-dir outputs/run_58day_full` | `.../run_58day_full/lanl_features.csv` |
| Red-team: filtering (needs `auth.txt.gz`) | `lanl_real_pipeline/cyber1_full/filter_auth.sh` | `filt.csv` (not distributed) |
| Red-team: label counts (Section 8.1; needs `redteam.txt.gz`) | `cyber1_full/redteam_summary.py` | `cyber1_full/results/redteam_summary.json` |
| Red-team: hourly series | `cyber1_full/cyber1_full.py` | `cyber1_full/results/hourly.csv` (its `cyber1_full_results.json` holds a post-campaign calibration that the article does not report; see the script's docstring) |
| Table 5, hourly AUCs and autocorrelation (Section 8.2) | `cyber1_full/cyber1_hourly_checks.py` | `cyber1_full/results/hourly_checks.json` |
| Hourly bound (Section 8.2) | `cyber1_full/entropy_bound.py` | `cyber1_full/results/entropy_bound.json` |
| 500-logon windows (Section 8.2) | `cyber1_full/cyber1_windows.py` | `cyber1_full/results/cyber1_windows_results.json`, `windows_rt.csv` |
| Table 6 (per-host scores) | `cyber1_full/cyber1_perhost.py`, then `cyber1_full/cyber1_budget_curve.py` and `cyber1_full/cyber1_novelty.py` | `cyber1_full/results/hostday_perhost.csv.gz`, `cyber1_budget_curve.json`, `cyber1_novelty.json` |
| Fig. 4 | `cyber1_full/cyber1_figure.py` | `figures/fig4_cyber1.*` |
| Section 4.5 (cost) | `code/benchmark_latency.py` | `data/benchmark_latency.json` |
| All tables | `code/make_tables.py` | `tables/*.tex` |
| Numbers quoted in the text | `code/check_claims.py` | printed (checked against `code/claims_text.txt`, the passages of the submitted manuscript that quote them) |

## Notes

- Synthetic experiments are seeded and reproduce exactly. The Random Forest
  in `enhanced_analysis.py` uses all cores, and its scores can differ in the
  third decimal across scikit-learn versions.
- Runtimes on a laptop: identifiability about 30 minutes, EKF about 10
  minutes, early warning about 2 hours; `regen_figs_final.py`,
  `lanl_extra_checks.py`, `make_tables.py` and `check_claims.py` under a
  minute each, `cyber1_novelty.py` about an hour (bootstrap).
- The red-team runs need about 30 GB of disk and took 38 minutes (filter)
  and about 1 hour per analysis script on one server core. The scripts
  after `cyber1_full.py` need only the files in `cyber1_full/results/`.
- `check_claims.py [main.tex]` compares the outputs with the manuscript
  when given one (or when `Computer_Communications_Submission/main.tex` is
  present), and otherwise with `code/claims_text.txt`.
