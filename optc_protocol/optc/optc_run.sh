#!/bin/bash
# Held-out OpTC evaluation after all option-A files are filtered:
#   build (dedup, mapping, host-units) -> Powell-style roles -> fixed-budget evaluation.
# Powell windows follow Amendment 1 (A2): 168 observed hours, 7 observed days.
# Usage: ./optc_run.sh FLOWS_DIR LABELS_DIR BUILD_DIR
set -euo pipefail
cd "$(dirname "$0")/.."
FLOWS=$1; LABELS=$2; B=$3
python3 optc/optc_build.py "$FLOWS" "$LABELS" "$B"
python3 powell_baseline.py "$B/edges_hour.csv.gz" hour "$B/powell_hour.csv.gz" --window 168
python3 powell_baseline.py "$B/edges_day.csv.gz" day "$B/powell_day.csv.gz" --window 7
python3 optc/optc_budget.py "$B" --out "$B/optc_budget.json"
