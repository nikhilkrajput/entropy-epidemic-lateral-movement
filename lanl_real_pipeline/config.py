"""
Configuration for the real-LANL entropy-epidemic pipeline.

==> THE ONLY THING YOU NORMALLY NEED TO CHANGE IS `LANL_DATA_PATH` BELOW. <==

Set it to wherever the LANL Unified Host and Network Data Set lives on your
machine, then run:

    python run_lanl_analysis.py

Everything else (file-format detection, filtering, windowing, figures) is
handled automatically.  Each knob is documented inline so you can override it
if your copy of the data is laid out differently.

Precedence (highest wins):  command-line flag  >  environment variable  >  this file.
  e.g.  python run_lanl_analysis.py --data-path "E:/lanl/wls"
  e.g.  set LANL_DATA_PATH=E:/lanl/wls   (Windows)   /   export LANL_DATA_PATH=...  (Linux/macOS)
"""

import os

# ---------------------------------------------------------------------------
# 1. WHERE IS YOUR DATA?  (set this)
# ---------------------------------------------------------------------------
# This may point to ANY of the following and the loader will figure it out:
#   * a directory holding the daily files, e.g. r"E:/lanl/wls"
#       (it will pick up wls_day-*.{csv,json,gz,bz2} automatically)
#   * a glob pattern, e.g. r"E:/lanl/wls/wls_day-*_2v.csv"
#   * a single file, e.g. r"E:/lanl/wls_day-01_2v.csv"
#   * a pre-processed auth CSV with columns
#       [epoch_time, src_host, dst_host, username]
#
# Leave it as None to fall back to the bundled tiny SAMPLE (smoke test only).
LANL_DATA_PATH = None  # <-- e.g. r"E:/datasets/lanl_unified_2017/wls"

# ---------------------------------------------------------------------------
# 2. FILE FORMAT  (usually leave on "auto")
# ---------------------------------------------------------------------------
#   "auto"         detect per-file from extension + content (recommended)
#   "wls_json"     LANL 2017 Unified host events, one JSON object per line
#   "wls_csv"      LANL 2017 host events exported to CSV WITH a header row
#   "wls_csv_raw"  LANL 2017 host events exported to CSV WITHOUT a header row
#                  (the native 21-column layout; src=Source col, dst=LogHost col)
#   "auth2015"     LANL 2015 "Comprehensive" auth.txt (headerless, 9 columns)
#   "preprocessed" a CSV already reduced to [epoch_time, src_host, dst_host, username]
DATA_FORMAT = "auto"

# ---------------------------------------------------------------------------
# 3. AUTHENTICATION FILTER  (matches the paper's methodology)
# ---------------------------------------------------------------------------
# Keep only the events that represent host-to-host lateral connectivity.
# Windows event 4624 = "an account was successfully logged on".
EVENT_IDS = {4624}
# Logon types: 3 = Network, 10 = RemoteInteractive (RDP).  These are the
# "network-logon and remote-interactive authentications" used in the paper.
LOGON_TYPES = {3, 10}
DROP_SELF_LOOPS = True          # discard events where src_host == dst_host
DROP_MACHINE_ACCOUNTS = False   # if True, drop usernames ending in '$' (computer accts)
REQUIRE_SUCCESS = True          # auth2015 only: keep success== "Success"
# Hosts to exclude entirely (case-insensitive). EMPTY by default, which keeps
# "ActiveDirectory" as a destination category -- matching the paper's analysis.
# For a PURE host-to-host graph (no domain-controller auths), set:
#     EXCLUDE_HOSTS = {"ActiveDirectory"}
EXCLUDE_HOSTS = set()

# ---------------------------------------------------------------------------
# 4. COLUMN MAPPING for custom CSVs  (only used if your headers are non-standard)
# ---------------------------------------------------------------------------
# The loader already understands the canonical LANL field names case-insensitively
# (Time/EventID/LogHost/Source/LogonType/UserName ...).  If your export uses
# different header names, map them here:  {"your_header": "canonical_name"}.
# Canonical names: time, eventid, loghost, source, logontype, username.
COLUMN_OVERRIDES = {}

# ---------------------------------------------------------------------------
# 5. SLIDING-WINDOW FEATURE EXTRACTION  (paper defaults; change with care)
# ---------------------------------------------------------------------------
WINDOW_FLOWS = 500     # events per observation window (W in the paper)
WINDOW_STRIDE = 250    # hop between consecutive windows (50% overlap)

# ---------------------------------------------------------------------------
# 6. EPIDEMIC FIT
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 7. INGEST BUDGET  (cap how much of each file is read; None = read everything)
# ---------------------------------------------------------------------------
# Main analysis (outputs/run_58day_full): first 48 MB of each daily file.
# Robustness check (outputs/run_58day): first 20 MB.  To reproduce, set
# MAX_BYTES_PER_FILE = 48 * 1024 * 1024 (or 20 * 1024 * 1024).  Budget is on the DECOMPRESSED
# byte stream and always stops on a clean line boundary.
MAX_BYTES_PER_FILE = None
MAX_ROWS_PER_FILE = None
MAX_EVENTS_TOTAL = None

# ---------------------------------------------------------------------------
# 8. OUTPUT
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_HERE, "outputs")
FIGURE_BASENAME = "fig_lanl_entropy_epidemic"   # local outputs/ filename (.pdf/.png)

# Optional: when set, a real-data run also copies the diagnostic figure into
# this folder under PAPER_FIGURE_BASENAME (the synthetic sample is never
# copied).  Off by default; override per run with --paper-figures-dir.
PAPER_FIGURES_DIR = None
PAPER_FIGURE_BASENAME = "fig_lanl_diagnostic"

RANDOM_SEED = 42


def resolve_data_path(cli_value=None):
    """CLI flag > env var > config.py value > bundled sample."""
    if cli_value:
        return cli_value
    env = os.environ.get("LANL_DATA_PATH")
    if env:
        return env
    if LANL_DATA_PATH:
        return LANL_DATA_PATH
    return os.path.join(_HERE, "sample_data")  # smoke-test fallback
