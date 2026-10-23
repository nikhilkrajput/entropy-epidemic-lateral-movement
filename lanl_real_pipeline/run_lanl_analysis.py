"""
Entry point: run the entropy-epidemic pipeline on the real LANL dataset.

USAGE
-----
1) Easiest -- edit LANL_DATA_PATH in config.py, then:
       python run_lanl_analysis.py

2) Or point it at the data on the command line (overrides config.py):
       python run_lanl_analysis.py --data-path "E:/datasets/lanl_unified_2017/wls"

3) Reproduce the article's subset (first 48 MB of each daily file):
       python run_lanl_analysis.py --data-path ... --max-bytes-per-file 50331648

4) Smoke test on the bundled tiny synthetic sample (no real data needed):
       python make_sample_data.py        # writes sample_data/wls_day-0X_2v.csv
       python run_lanl_analysis.py       # auto-uses sample_data/ if path unset

Outputs land in config.OUTPUT_DIR (default: ./outputs/):
    lanl_features.csv                 per-window entropy features
    lanl_results.json                 linkage r, growth fit, RF R^2, ...
    fig_lanl_entropy_epidemic.pdf/png a 2-panel diagnostic figure (not in the article)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402
import lanl_loader
import lanl_pipeline


def main():
    ap = argparse.ArgumentParser(description="Real-LANL entropy-epidemic pipeline")
    ap.add_argument("--data-path", default=None,
                    help="dir | glob | file for the LANL data (overrides config.py)")
    ap.add_argument("--format", default=None,
                    choices=["auto", "wls_json", "wls_csv", "wls_csv_raw",
                             "auth2015", "preprocessed"],
                    help="force a file format (default: config.DATA_FORMAT)")
    ap.add_argument("--output-dir", default=None, help="override config.OUTPUT_DIR")
    ap.add_argument("--max-bytes-per-file", type=int, default=None,
                    help="cap decompressed bytes read per file (e.g. 20971520 = 20 MB)")
    ap.add_argument("--max-rows-per-file", type=int, default=None)
    ap.add_argument("--max-events-total", type=int, default=None)
    ap.add_argument("--paper-figures-dir", default=None,
                    help="also copy the diagnostic figure into this dir (overrides config)")
    ap.add_argument("--no-paper-figure", action="store_true",
                    help="do NOT copy the figure into the paper's figures/ dir")
    args = ap.parse_args()

    if args.format:
        cfg.DATA_FORMAT = args.format
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.max_bytes_per_file is not None:
        cfg.MAX_BYTES_PER_FILE = args.max_bytes_per_file
    if args.max_rows_per_file is not None:
        cfg.MAX_ROWS_PER_FILE = args.max_rows_per_file
    if args.max_events_total is not None:
        cfg.MAX_EVENTS_TOTAL = args.max_events_total
    if args.no_paper_figure:
        cfg.PAPER_FIGURES_DIR = None
    elif args.paper_figures_dir:
        cfg.PAPER_FIGURES_DIR = args.paper_figures_dir

    data_path = cfg.resolve_data_path(args.data_path)
    using_sample = data_path.rstrip("/\\").endswith("sample_data")

    print("=" * 70)
    print("Real-LANL Entropy-Epidemic Pipeline")
    print("=" * 70)
    print(f"Data path : {data_path}")
    print(f"Format    : {cfg.DATA_FORMAT}")
    print(f"Output    : {cfg.OUTPUT_DIR}")
    if cfg.PAPER_FIGURES_DIR and not using_sample:
        paper_base = getattr(cfg, "PAPER_FIGURE_BASENAME", cfg.FIGURE_BASENAME)
        print(f"Paper fig : {os.path.join(cfg.PAPER_FIGURES_DIR, paper_base + '.pdf')}")
    elif cfg.PAPER_FIGURES_DIR and using_sample:
        print("Paper fig : (skipped for synthetic sample)")
    if using_sample:
        print("\n*** NOTE: using the bundled SYNTHETIC sample (smoke test only). ***")
        print("*** Set LANL_DATA_PATH in config.py to analyse the real dataset. ***")
    print()

    events = lanl_loader.load_events(cfg, data_path)
    results = lanl_pipeline.run(events, cfg, data_path_label=data_path,
                                is_sample=using_sample)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    link = results["linkage_calibrated_on_real_data"]
    fit = results["logistic_growth_fit"]
    rf = results["real_data_random_forest"]
    print(f"  events / hosts        : {results['n_events_total']:,} / {results['n_hosts_network']:,}")
    print(f"  windows               : {results['n_windows']:,}")
    print(f"  linkage Pearson r     : {link['pearson_r']:.4f} (p={link['pearson_p_value']:.2e})")
    print(f"  linkage H0, eta       : {link['H0_bits']:.3f} bits, {link['eta_bits_per_iota']:.2f}")
    print(f"  logistic growth rate  : {fit['r_per_day']:.3f}/day" if fit['r_per_day'] == fit['r_per_day']
          else "  logistic growth rate  : n/a (need >=4 days)")
    if rf.get("R2_random_kfold_mean") is not None:
        print(f"  RF R2 (random k-fold) : {rf['R2_random_kfold_mean']:.4f}")
    if rf.get("R2_leave_one_day_out_mean") is not None:
        print(f"  RF R2 (day-out CV)    : {rf['R2_leave_one_day_out_mean']:.4f}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
