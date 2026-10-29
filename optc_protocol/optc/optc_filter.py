"""
Stage 1 for OpTC: keep the internal FLOW START events of one eCAR file.

Reads decompressed eCAR JSON lines on stdin (pigz -dc file | python3 ...),
applies the selection of LMDEval's extract_optc.py (Larroche et al.):
FLOW START events whose source and destination addresses are both internal
(10.x except .255, 142.x except 142.20.59.255, fe80:), and writes one row
per event to a Parquet file:
  ts       seconds since 1970-01-01T00:00:00-04:00 (local exercise time)
  host     reporting hostname
  sip, dip, sport, dport, proto, outbound (1 if direction == outbound)
  lab      2 = LMDEval 'Lateral movement' id, 1 = other red-team id, 0 = none
Deduplication of flows seen at both ends is done later, over all files
(optc_build.py), because the two ends are in different files.

Usage: pigz -dc X.json.gz | python3 optc_filter.py optc_redteam.csv out.parquet
"""
import calendar
import csv
import datetime as dt
import json
import sys

import pyarrow as pa
import pyarrow.parquet as pq

LOCAL_OFFSET = 4 * 3600   # ts origin 1970-01-01T00:00:00-04:00, as in LMDEval
_day0 = {}


def parse_ts(s):
    """'2019-09-25T09:05:37.29-04:00' -> seconds since 1970-01-01T00:00:00-04:00.
    Python 3.10's fromisoformat rejects fractions that are not 3 or 6 digits."""
    d = s[:10]
    base = _day0.get(d)
    if base is None:
        base = _day0[d] = calendar.timegm(dt.date.fromisoformat(d).timetuple())
    sec = int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19])
    rest = s[19:]
    k = max(rest.rfind("+"), rest.rfind("-"))
    frac = float("0" + rest[:k]) if k > 0 and rest[0] == "." else 0.0
    tz = rest[k:] if k >= 0 else "+00:00"
    off = (1 if tz[0] == "+" else -1) * (int(tz[1:3]) * 3600 + int(tz[4:6]) * 60)
    return base + sec + frac - off - LOCAL_OFFSET


def include_ip(a):
    if a.startswith("10.") and not a.endswith(".255"):
        return True
    if a.startswith("142.") and a != "142.20.59.255":
        return True
    return a.startswith("fe80:")


labels = {}
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        labels[r["id"]] = 2 if r["label"] == "Lateral movement" else 1

cols = {k: [] for k in ("ts", "host", "sip", "dip", "sport", "dport", "proto", "outbound", "lab")}
n_lines = n_flow = 0
for line in sys.stdin:
    n_lines += 1
    if line[1:17] != '"action":"START"' or '"object":"FLOW"' not in line:
        continue
    e = json.loads(line)
    p = e["properties"]
    sip, dip = p["src_ip"], p["dest_ip"]
    if not (include_ip(sip) and include_ip(dip)):
        continue
    n_flow += 1
    cols["ts"].append(parse_ts(e["timestamp"]))
    cols["host"].append(e["hostname"])
    cols["sip"].append(sip)
    cols["dip"].append(dip)
    cols["sport"].append(int(p["src_port"]))
    cols["dport"].append(int(p["dest_port"]))
    cols["proto"].append(int(p["l4protocol"]))
    cols["outbound"].append(1 if p["direction"] == "outbound" else 0)
    cols["lab"].append(labels.get(e["id"], 0))

tab = pa.table({"ts": pa.array(cols["ts"], pa.float64()), "host": pa.array(cols["host"]).dictionary_encode(),
                "sip": pa.array(cols["sip"]).dictionary_encode(), "dip": pa.array(cols["dip"]).dictionary_encode(),
                "sport": pa.array(cols["sport"], pa.int32()), "dport": pa.array(cols["dport"], pa.int32()),
                "proto": pa.array(cols["proto"], pa.int16()), "outbound": pa.array(cols["outbound"], pa.int8()),
                "lab": pa.array(cols["lab"], pa.int8())})
pq.write_table(tab, sys.argv[2], compression="zstd")
print(f"lines {n_lines} internal_flow_start {n_flow} labelled {sum(1 for x in cols['lab'] if x)}", file=sys.stderr)
