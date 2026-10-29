# Pre-registered protocol for the OpTC held-out evaluation

This folder freezes the evaluation of per-host lateral-movement scores on the DARPA OpTC data. It was published
**before any red-team-day OpTC record had been scored**. The tag `optc-prereg-v1` marks that state.

## Protocol files
- `PREREGISTRATION.md` is the protocol. It was written after the detector had been developed on LANL 2015 and before
  any OpTC flow was read.
- `PREREGISTRATION_AMENDMENT_1.md` resolves points where the protocol text and the code disagreed. It was written
  before any red-team-day record was read.
- `preregistration_sha256.txt` holds the SHA-256 of both files, with the local time each was written.

## Code
- `budget_eval.py` contains the fixed-budget metrics, the first-contact surprise score and the novelty controls.
- `hostunit.py` computes the per-host-unit scores.
- `powell_baseline.py` is the graph-role comparator.
- `stream_detector.py` is the streaming implementation of the detector.
- `lanl_budget.py` is the LANL development analysis that fixed the prior weight a.
- `optc/` covers:
  - fetching and filtering the OpTC eCAR files (`optc_fetch.sh`, `optc_filter.py`, with the file list
    `optc_files_optionA.tsv`);
  - building host units (`optc_build.py`);
  - the evaluation (`optc_budget.py`), which refuses to run if either protocol file's hash has changed;
  - the runner (`optc_run.sh`).

## Labels
The labels and the preprocessing conventions come from LMDEval (https://github.com/cl-anssi/LMDEval, commit 2b95a68),
which is not redistributed here.
