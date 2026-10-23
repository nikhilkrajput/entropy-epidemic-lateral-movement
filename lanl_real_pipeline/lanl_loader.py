"""
Robust loader for the LANL Unified Host and Network Data Set.

Goal: given ONE path (directory, glob, or single file), return a tidy
DataFrame of host-to-host authentication events with exactly four columns:

    epoch_time : int    seconds (relative to the dataset's own origin)
    src_host   : str    originating computer
    dst_host   : str    destination computer (where the logon was recorded)
    username   : str

It transparently handles, and auto-detects between:
    * LANL 2017 host events as JSON-lines        (wls_day-XX.json / .bz2 / .gz)
    * LANL 2017 host events exported to CSV       (wls_day-XX_2v.csv)
    * LANL 2015 "Comprehensive" auth.txt          (headerless, 9 columns)
    * an already-reduced CSV [epoch_time, src_host, dst_host, username]

Compression (.gz, .bz2) is detected from the extension. A per-file byte/row
budget can be applied (see config.MAX_BYTES_PER_FILE) and always stops on a
clean line boundary.
"""

import bz2
import glob
import gzip
import io
import json
import os

import pandas as pd

UNIFIED_COLS = ["epoch_time", "src_host", "dst_host", "username"]

# Canonical LANL host-event field names (lower-cased) we know how to consume.
_CANON = {
    "time": "time",
    "eventid": "eventid",
    "loghost": "loghost",
    "source": "source",
    "logontype": "logontype",
    "username": "username",
    "targetusername": "targetusername",
    "subjectusername": "subjectusername",
    "destination": "destination",
}

_AUTH2015_COLS = [
    "time", "source_user", "destination_user", "source_computer",
    "destination_computer", "auth_type", "logon_type",
    "auth_orientation", "success",
]

# LANL 2017 Unified host-event (wls) "_2v" export WITHOUT a header row: 21
# columns in the official fixed order (per the dataset's own schema):
# col 2 = log_host, col 12 = src_host, col 13 = src_user, col 14 = dst_host.
# We map the lateral-movement graph from the explicit src_host/dst_host columns.
_WLS_RAW_COLS = [
    "time", "eventid", "loghost", "logontype", "logontypedesc", "username",
    "domainname", "logonid", "subjectusername", "subjectdomainname",
    "subjectlogonid", "status", "src_host", "src_user", "dst_host",
    "authpackage", "failurereason", "processname", "processid",
    "parentprocessname", "parentprocessid",
]

# Textual logon-type names -> the numeric codes used in config.LOGON_TYPES.
_LOGON_NAME_TO_CODE = {
    "interactive": 2, "network": 3, "batch": 4, "service": 5, "unlock": 7,
    "networkcleartext": 8, "newcredentials": 9, "remoteinteractive": 10,
    "cachedinteractive": 11,
}

# Genuinely empty / null markers that are never a real host. NOTE we do NOT
# put "activedirectory" here: in the LANL data it is the (anonymised) name of
# the domain-controller destination and the paper's analysis treats it as a
# regular host category. Drop it only if you opt in via config.EXCLUDE_HOSTS.
_PLACEHOLDER_HOSTS = {"", "-", "?", "none", "null", "nan"}


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def resolve_files(path):
    """Expand `path` (dir | glob | file) into a sorted list of files."""
    if path is None:
        raise FileNotFoundError("No data path configured. Set LANL_DATA_PATH in config.py.")
    if any(ch in path for ch in "*?[") and not os.path.isdir(path):
        files = sorted(glob.glob(path))
    elif os.path.isdir(path):
        files = []
        for pat in ("wls_day-*", "*.csv", "*.json", "*.jsonl",
                    "*.txt", "*.gz", "*.bz2"):
            files.extend(glob.glob(os.path.join(path, pat)))
        files = sorted(set(files))
    elif os.path.isfile(path):
        files = [path]
    else:
        raise FileNotFoundError(f"Data path does not exist: {path}")
    if not files:
        raise FileNotFoundError(f"No data files found under: {path}")
    return files


# ---------------------------------------------------------------------------
# Low-level reading (handles compression + budget, never cuts a line)
# ---------------------------------------------------------------------------
def _open_binary(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    if path.endswith(".bz2"):
        return bz2.open(path, "rb")
    return open(path, "rb")


def read_text_lines(path, max_bytes=None, max_rows=None, encoding="utf-8"):
    """Yield decoded lines from `path`, honouring optional byte/row budgets.

    Iterating a binary handle yields complete '\\n'-terminated chunks, so the
    budget can never truncate a line mid-way. Budget is measured on the
    DECOMPRESSED stream.
    """
    n_bytes = 0
    n_rows = 0
    with _open_binary(path) as fh:
        for raw in fh:
            yield raw.decode(encoding, errors="replace")
            n_bytes += len(raw)
            n_rows += 1
            if max_bytes is not None and n_bytes >= max_bytes:
                break
            if max_rows is not None and n_rows >= max_rows:
                break


def _peek_lines(path, n=5):
    out = []
    for i, line in enumerate(read_text_lines(path)):
        if line.strip():
            out.append(line)
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------
def detect_format(path):
    """Return one of: wls_json | wls_csv | auth2015 | preprocessed."""
    low = path.lower()
    peek = _peek_lines(path, n=3)
    first = peek[0].strip() if peek else ""

    if first.startswith("{"):
        return "wls_json"
    if low.endswith((".json", ".jsonl")):
        return "wls_json"

    header = first.lower()
    tokens = [t.strip().strip('"') for t in header.split(",")]
    tokenset = set(tokens)

    if {"src_host", "dst_host"}.issubset(tokenset) or \
       {"epoch_time", "src_host"}.issubset(tokenset):
        return "preprocessed"
    if "eventid" in tokenset or {"loghost", "logontype"}.issubset(tokenset) \
       or {"loghost", "source"}.issubset(tokenset):
        return "wls_csv"

    # Headerless raw LANL 2017 wls export: ~21 comma fields and field[1] is a
    # 4-digit Windows EventID (e.g. 4624). First line is data, not a header.
    raw_tokens = [t.strip() for t in first.split(",")]
    if len(raw_tokens) >= 15 and len(raw_tokens) > 1 \
       and raw_tokens[1].isdigit() and len(raw_tokens[1]) == 4:
        return "wls_csv_raw"

    # Headerless 9-field rows with computer-like tokens -> 2015 auth.txt.
    has_header_word = any(
        any(c.isalpha() for c in t) and t in _CANON for t in tokenset)
    if len(tokens) >= 9 and not has_header_word:
        return "auth2015"

    return "preprocessed"


# ---------------------------------------------------------------------------
# Per-format parsing -> raw DataFrame
# ---------------------------------------------------------------------------
def _read_csv_budgeted(path, max_bytes, max_rows, header="infer", names=None):
    if max_bytes is None and max_rows is None:
        return pd.read_csv(path, compression="infer", header=header,
                           names=names, dtype=str, low_memory=False)
    text = "".join(read_text_lines(path, max_bytes=max_bytes, max_rows=max_rows))
    return pd.read_csv(io.StringIO(text), header=header, names=names, dtype=str)


def _parse_wls_json(path, max_bytes, max_rows):
    records = []
    for line in read_text_lines(path, max_bytes=max_bytes, max_rows=max_rows):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return pd.DataFrame.from_records(records)


def _parse(path, fmt, cfg):
    mb, mr = cfg.MAX_BYTES_PER_FILE, cfg.MAX_ROWS_PER_FILE
    if fmt == "wls_json":
        return _parse_wls_json(path, mb, mr)
    if fmt == "auth2015":
        return _read_csv_budgeted(path, mb, mr, header=None, names=_AUTH2015_COLS)
    if fmt == "wls_csv_raw":
        return _read_csv_budgeted(path, mb, mr, header=None, names=_WLS_RAW_COLS)
    # wls_csv and preprocessed both have headers
    return _read_csv_budgeted(path, mb, mr, header="infer")


# ---------------------------------------------------------------------------
# Normalisation -> unified [epoch_time, src_host, dst_host, username]
# ---------------------------------------------------------------------------
def _canon_columns(df, overrides):
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if col in overrides:
            rename[col] = overrides[col]
        elif key in overrides:
            rename[col] = overrides[key]
        elif key in _CANON:
            rename[col] = _CANON[key]
        else:
            rename[col] = key
    out = df.rename(columns=rename)
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def _to_int_series(s):
    return pd.to_numeric(s, errors="coerce")


def _clean_host(series):
    # fillna("") FIRST: empty CSV fields read as NaN, and pandas>=3.0 keeps NaN
    # through astype(str) (it does not become the literal "nan"), which would
    # leave float NaN mixed with strings and break np.unique downstream.
    return series.fillna("").astype(str).str.strip()


def _is_real_host(series):
    low = series.fillna("").astype(str).str.strip().str.lower()
    return ~low.isin(_PLACEHOLDER_HOSTS)


def _logon_matches(series, wanted_codes):
    """Match logon type whether stored as a number or a textual name."""
    num = pd.to_numeric(series, errors="coerce")
    by_num = num.isin(list(wanted_codes))
    names = series.astype(str).str.strip().str.lower()
    name_codes = names.map(_LOGON_NAME_TO_CODE)
    by_name = name_codes.isin(list(wanted_codes))
    return by_num.fillna(False) | by_name.fillna(False)


def normalize_and_filter(df, fmt, cfg):
    if df.empty:
        return pd.DataFrame(columns=UNIFIED_COLS)

    if fmt == "preprocessed":
        d = _canon_columns(df, cfg.COLUMN_OVERRIDES)
        colmap = {}
        for cand in ("epoch_time", "time"):
            if cand in d.columns:
                colmap["epoch_time"] = cand
                break
        for cand in ("src_host", "source", "source_computer"):
            if cand in d.columns:
                colmap["src_host"] = cand
                break
        for cand in ("dst_host", "loghost", "destination", "destination_computer"):
            if cand in d.columns:
                colmap["dst_host"] = cand
                break
        for cand in ("username", "user", "source_user"):
            if cand in d.columns:
                colmap["username"] = cand
                break
        missing = {"epoch_time", "src_host", "dst_host"} - set(colmap)
        if missing:
            raise ValueError(
                f"Pre-processed file is missing columns {missing}. "
                f"Use config.COLUMN_OVERRIDES to map them.")
        out = pd.DataFrame({
            "epoch_time": _to_int_series(d[colmap["epoch_time"]]),
            "src_host": _clean_host(d[colmap["src_host"]]),
            "dst_host": _clean_host(d[colmap["dst_host"]]),
            "username": d[colmap["username"]].astype(str) if "username" in colmap
            else "unknown",
        })

    elif fmt == "auth2015":
        d = df.copy()
        keep = pd.Series(True, index=d.index)
        if cfg.REQUIRE_SUCCESS and "success" in d.columns:
            keep &= d["success"].astype(str).str.strip().str.lower().eq("success")
        if "auth_orientation" in d.columns:
            keep &= d["auth_orientation"].astype(str).str.strip().str.lower().eq("logon")
        keep &= _logon_matches(d["logon_type"], cfg.LOGON_TYPES)
        d = d[keep]
        user = d["source_user"].astype(str).str.split("@").str[0]
        out = pd.DataFrame({
            "epoch_time": _to_int_series(d["time"]),
            "src_host": _clean_host(d["source_computer"]),
            "dst_host": _clean_host(d["destination_computer"]),
            "username": user,
        })

    else:  # wls_json / wls_csv / wls_csv_raw
        d = _canon_columns(df, cfg.COLUMN_OVERRIDES)
        # Resolve src/dst host columns: prefer the explicit src_host/dst_host
        # of the LANL 2017 "_2v" export; fall back to Source/LogHost as used in
        # the host-event JSON form.
        src_col = next((c for c in ("src_host", "source") if c in d.columns), None)
        dst_col = next((c for c in ("dst_host", "loghost", "destination")
                        if c in d.columns), None)
        if src_col is None or dst_col is None:
            raise ValueError(
                f"wls file lacks a source/destination host field; got columns "
                f"{list(d.columns)[:14]}... Use COLUMN_OVERRIDES to map them.")
        keep = pd.Series(True, index=d.index)
        if "eventid" in d.columns:
            keep &= _to_int_series(d["eventid"]).isin(list(cfg.EVENT_IDS))
        if "logontype" in d.columns:
            keep &= _logon_matches(d["logontype"], cfg.LOGON_TYPES)
        d = d[keep]
        user_col = next((c for c in ("username", "targetusername",
                                     "subjectusername", "src_user")
                         if c in d.columns), None)
        user = (d[user_col].fillna("").astype(str) if user_col
                else pd.Series("unknown", index=d.index))
        time_col = "time" if "time" in d.columns else (
            "epoch_time" if "epoch_time" in d.columns else None)
        out = pd.DataFrame({
            "epoch_time": _to_int_series(d[time_col]),
            "src_host": _clean_host(d[src_col]),
            "dst_host": _clean_host(d[dst_col]),
            "username": user,
        })

    # Common cleaning for every format.
    out = out.dropna(subset=["epoch_time"])
    out = out[_is_real_host(out["src_host"]) & _is_real_host(out["dst_host"])]
    exclude = {h.strip().lower() for h in getattr(cfg, "EXCLUDE_HOSTS", set())}
    if exclude:
        keep = ~(out["src_host"].str.lower().isin(exclude)
                 | out["dst_host"].str.lower().isin(exclude))
        out = out[keep]
    if cfg.DROP_SELF_LOOPS:
        out = out[out["src_host"] != out["dst_host"]]
    if cfg.DROP_MACHINE_ACCOUNTS:
        out = out[~out["username"].astype(str).str.endswith("$")]
    out["epoch_time"] = out["epoch_time"].astype("int64")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def load_events(cfg, data_path):
    """Load + filter every file under `data_path` into one unified DataFrame."""
    files = resolve_files(data_path)
    print(f"Discovered {len(files)} file(s) under {data_path}")
    frames = []
    total = 0
    for fp in files:
        fmt = detect_format(fp) if cfg.DATA_FORMAT == "auto" else cfg.DATA_FORMAT
        raw = _parse(fp, fmt, cfg)
        tidy = normalize_and_filter(raw, fmt, cfg)
        print(f"  {os.path.basename(fp):40s} [{fmt:12s}] "
              f"{len(raw):>10,} rows -> {len(tidy):>9,} auth events")
        if len(tidy):
            frames.append(tidy)
            total += len(tidy)
        if cfg.MAX_EVENTS_TOTAL is not None and total >= cfg.MAX_EVENTS_TOTAL:
            break

    if not frames:
        raise RuntimeError(
            "No authentication events survived filtering. Check DATA_FORMAT, "
            "EVENT_IDS, LOGON_TYPES, or COLUMN_OVERRIDES in config.py.")

    events = pd.concat(frames, ignore_index=True)
    if cfg.MAX_EVENTS_TOTAL is not None:
        events = events.iloc[:cfg.MAX_EVENTS_TOTAL]
    events = events.sort_values("epoch_time").reset_index(drop=True)
    print(f"Unified total: {len(events):,} auth events, "
          f"{events['src_host'].nunique():,} src / "
          f"{events['dst_host'].nunique():,} dst hosts")
    return events
