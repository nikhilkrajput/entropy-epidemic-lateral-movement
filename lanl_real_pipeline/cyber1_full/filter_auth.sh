#!/bin/bash
# Stage 1: stream LANL cyber1 auth.txt.gz and keep successful LogOn events
# between distinct computers.  Output: time,src_computer,dst_computer,logon_type
# Usage: ./filter_auth.sh auth.txt.gz filt.csv
set -euo pipefail
pigz -dc -p 4 "$1" \
  | mawk -F, 'BEGIN{OFS=","} $8=="LogOn" && $9=="Success" && $4!=$5 {print $1,$4,$5,$7}' \
  > "$2"
wc -l "$2"
