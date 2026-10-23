# Entropy observables for epidemic models of cyber lateral movement

Code and derived outputs for the article

> S. Verma, N. K. Rajput, *Observing Epidemic Spreading Through Traffic
> Entropy: Detectability Limits, Early Warning and Real Lateral Movement in
> Enterprise Networks* (manuscript under review).

Every number, table and figure in the article is produced by a script in
this repository. `REPRODUCE.md` maps each one to its script and output file.

## Contents

| Folder | What it holds |
|---|---|
| `code/` | Logon-traffic simulator, the synthetic experiments (identifiability, early warning, EKF, detectability bound, latency), and the scripts that build the tables and check the text |
| `data/` | Outputs of the synthetic experiments (JSON) |
| `lanl_real_pipeline/` | Pipeline for the LANL Unified Host and Network data (2017): loading, window features, linkage, regression |
| `lanl_real_pipeline/outputs/` | Cached window features and results for the 58-day subset (`run_58day_full`, 48 MB daily prefixes) and the 20 MB prefix check (`run_58day`) |
| `lanl_real_pipeline/cyber1_full/` | Full-campaign evaluation on the LANL red-team data (2015): filter, network-wide detectors, fine windows, per-host scores |
| `lanl_real_pipeline/cyber1_full/results/` | Cached outputs of that evaluation |
| `figures/` | Figures as used in the article (PDF and PNG) |

## Data

The raw LANL data are **not** redistributed here. Obtain them from Los
Alamos National Laboratory:

- **Unified Host and Network Data Set (2017):** Turcotte, Kent and Hash,
  *Unified Host and Network Data Set*, in *Data Science for Cyber-Security*,
  World Scientific, 2018, doi:10.1142/9781786345646_001.
  `lanl_real_pipeline/download_lanl.py` fetches daily prefixes of the
  `wls_day-XX_2v.csv` files.
- **Comprehensive, Multi-Source Cyber-Security Events (2015, "cyber1"):**
  A. D. Kent, Los Alamos National Laboratory, 2015, doi:10.17021/1179829.
  Request `auth.txt.gz` and `redteam.txt.gz` from
  <https://csr.lanl.gov/data/cyber1/>.

The cached outputs contain only per-window and per-host-day statistics
derived from these data, with LANL's de-identified host names reduced to
their numeric part.

`lanl_real_pipeline/sample_data/` contains **synthetic** files, made by
`make_sample_data.py`, for testing the pipeline without the real data.

## Environment

- Synthetic experiments and LANL 2017 statistics:
  Python 3.12, `pip install -r requirements.txt`.
- Red-team evaluation (run on a Linux server):
  Python 3.10, `pip install -r requirements-redteam.txt` (NumPy, pandas,
  SciPy, scikit-learn), plus `pigz` and `mawk`.

## Quick start

```bash
pip install -r requirements.txt
python code/identifiability_traffic.py      # Table 1
python code/ekf_estimation.py               # Table 2, Fig. 1
python code/early_warning_traffic.py        # Table 3, Fig. 2
python code/detectability_bound.py          # Proposition 1 checks
python lanl_real_pipeline/regen_figs_final.py      # Fig. 3, growth numbers
python lanl_real_pipeline/lanl_extra_checks.py     # Section 7 checks
python code/make_tables.py                  # all tables, from the outputs
python code/check_claims.py                 # every number quoted in the text
```

Red-team evaluation (needs the 7.6 GB `auth.txt.gz`):

```bash
cd lanl_real_pipeline/cyber1_full
./filter_auth.sh auth.txt.gz filt.csv
python3 cyber1_full.py filt.csv redteam.txt.gz       # hourly detectors
python3 cyber1_windows.py filt.csv redteam.txt.gz    # 500-event windows
python3 cyber1_perhost.py filt.csv redteam.txt.gz    # per-host scores
python3 redteam_summary.py redteam.txt.gz             # label counts
python3 entropy_bound.py && python3 cyber1_hourly_checks.py
python3 cyber1_budget_curve.py && python3 cyber1_novelty.py
python3 cyber1_figure.py
```

## Licence

Code: MIT licence (`LICENSE`). The derived outputs in `data/` and the
`results/` and `outputs/` folders are statistics computed from LANL data
and are provided for reproducing the article; cite the LANL datasets
listed above when using them.

## Citation

If you use this code, please cite the article above and this repository,
<https://github.com/nikhilkrajput/entropy-epidemic-lateral-movement>.
