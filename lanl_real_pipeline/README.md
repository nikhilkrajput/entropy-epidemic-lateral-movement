# LANL pipelines

Code for the article's analyses of two LANL datasets:

- the **Unified Host and Network Data Set (2017)**, Section 7 of the
  article: this folder;
- the **Comprehensive, Multi-Source Cyber-Security Events data (2015)**
  with red-team labels, Section 8: `cyber1_full/` (see the repository's
  `REPRODUCE.md`).

The raw data are not included; obtain them from Los Alamos National
Laboratory (see the repository's `README.md`).

## LANL 2017: from raw data

Set the data path in `config.py` (`LANL_DATA_PATH`), pass it with
`--data-path`, or set the `LANL_DATA_PATH` environment variable. The path
may be a directory of `wls_day-*` files, a glob, a single file, or a
pre-processed CSV with columns `epoch_time, src_host, dst_host, username`.

The article uses the first 48 MB of each daily file:

```bash
python run_lanl_analysis.py --data-path <wls folder> --max-bytes-per-file 50331648 \
    --output-dir outputs/run_58day_full
```

This writes

| File | Contents |
|---|---|
| `lanl_features.csv` | per-window entropies, active-destination fraction `iota`, cumulative reach `iota_cum` |
| `lanl_results.json` | linkage fit and correlations, logistic fit of cumulative reach (growth rate only), Random Forest CV |
| `fig_lanl_entropy_epidemic.*` | a diagnostic figure (not used in the article) |

`outputs/run_58day/` holds the same outputs for 20 MB prefixes, a check
of the prefix length that the article does not report.

## LANL 2017: from the cached outputs (no raw data needed)

```bash
python regen_figs_final.py     # Fig. 3 and outputs/run_58day_full/growth.json
python lanl_extra_checks.py    # support size, next-window prediction (Section 7)
python enhanced_analysis.py    # regressor comparison (model_comparison.json)
```

## What the pipeline computes

1. Per-window active-destination fraction `iota = k/N` (`k` distinct
   destinations in the window) and cumulative reach `iota_cum`.
2. Windows of `W = 500` events, a new one every 250 events (`config.py`).
3. Six entropies per window: Shannon of source, destination and user name,
   Rényi-2 and Tsallis-2 of source, spectral entropy of inter-arrival times.
4. Least-squares fit `H_dst = H0 + eta * iota` and Pearson `r` for every
   entropy against `iota` and `iota_cum`.
5. Logistic fit of daily cumulative reach (growth rate `r` and plateau).
6. Random Forest `entropy -> iota` under random k-fold and
   leave-one-day-out CV.

Filtering (`config.py`): `EVENT_IDS={4624}`, `LOGON_TYPES={3,10}`
(network and remote-interactive), self-loops dropped.

## Smoke test (no real data)

```bash
python make_sample_data.py     # synthetic fixture in sample_data/
python run_lanl_analysis.py    # uses sample_data/ when no path is set
```

The fixture is synthetic and its numbers are meaningless.

## Files

```
config.py              data path and settings
lanl_loader.py         raw wls/auth/csv -> [epoch_time, src_host, dst_host, username]
lanl_pipeline.py       window entropies, linkage, growth fit, Random Forest, figure
run_lanl_analysis.py   entry point (command-line flags override config.py)
regen_figs_final.py    Fig. 3 from cached outputs
lanl_extra_checks.py   Section 7 checks from cached outputs
enhanced_analysis.py   regressor comparison from cached outputs
download_lanl.py       fetch daily prefixes of the wls files
make_sample_data.py    synthetic fixture (smoke test only)
cyber1_full/           LANL 2015 red-team evaluation
```
