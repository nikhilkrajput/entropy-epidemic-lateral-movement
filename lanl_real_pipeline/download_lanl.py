"""
Download LANL 2017 Unified host-event daily files via HTTP range requests.

This reproduces exactly how the 58-day subset used in the paper was obtained:
the first N megabytes of each daily `wls_day-XX_2v.csv` from the public mirror.
Dependency-free (stdlib urllib). Configure with flags or edit the defaults.

Examples
--------
    # first 48 MB of days 1..58 (the paper's main 58-day subset)
    python download_lanl.py --days 1-58 --mb 48 --out data_58day_full
    # first 20 MB (the paper's prefix-length robustness check)
    python download_lanl.py --days 1-58 --mb 20 --out data_58day

    # full days (WARNING: 3.9-6.2 GB each) -- only if you have the disk/time
    python download_lanl.py --days 1-7 --mb 6200 --out data_full

Then analyse with:
    python run_lanl_analysis.py --data-path data_58day
"""

import argparse
import os
import urllib.request

# Mirror documented in article_2_optc_calibration/data/DATASET_INFO.txt.
# Days available: 01..90. Each full file is several GB; range requests let us
# take just a prefix, matching the paper's 48 MB (main) / 20 MB (robustness) daily prefixes.
BASE_URL = "https://datasets.rocketgraph.com/LANL/xgt/wls_day-{:02d}_2v.csv"


def parse_days(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def download_one(day, n_bytes, out_dir, base_url=BASE_URL, timeout=180):
    url = base_url.format(day)
    dst = os.path.join(out_dir, f"wls_day-{day:02d}_2v.csv")
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{n_bytes - 1}"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dst, "wb") as f:
        f.write(r.read())
    return dst, os.path.getsize(dst)


def main():
    ap = argparse.ArgumentParser(description="Range-download LANL daily files")
    ap.add_argument("--days", default="1-58", help="e.g. '1-58' or '1,2,3' or '8-20'")
    ap.add_argument("--mb", type=int, default=20, help="megabytes to fetch per file")
    ap.add_argument("--out", default="data_58day", help="output directory")
    ap.add_argument("--base-url", default=BASE_URL,
                    help="URL template with a {:02d} day placeholder")
    args = ap.parse_args()

    days = parse_days(args.days)
    n_bytes = args.mb * 1024 * 1024
    os.makedirs(args.out, exist_ok=True)
    print(f"Downloading {len(days)} files x {args.mb} MB -> {args.out}/")

    ok, fail = 0, []
    for d in days:
        try:
            path, size = download_one(d, n_bytes, args.out, base_url=args.base_url)
            print(f"  day {d:02d}: {size:,} B -> {os.path.basename(path)}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  day {d:02d}: FAILED ({e})")
            fail.append(d)
    print(f"\nDone: {ok} ok, {len(fail)} failed" + (f" {fail}" if fail else ""))
    if not fail:
        print(f"Now run:  python run_lanl_analysis.py --data-path {args.out}")


if __name__ == "__main__":
    main()
