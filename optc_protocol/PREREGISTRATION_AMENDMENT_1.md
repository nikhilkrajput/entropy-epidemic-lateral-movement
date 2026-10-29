# Amendment 1 to the OpTC protocol

**Status when this amendment was written.** Written 2026-10-29 at about 14:15 UTC. By then only these OpTC files had
been filtered:
- five files of benign/19Sep19;
- one small file of evaluation/25Sept, which holds 128 internal flows and none of them labelled.

No red-team-day (23 or 24 Sep) flow had been read, and no score had been computed on evaluation data.

**Why it exists.** An internal review found four places where the frozen protocol (PREREGISTRATION.md, SHA-256
81184c43…1df6) and the code disagree, and two metrics that were defined in a misleading way. This amendment
resolves them before any evaluation data are scored. PREREGISTRATION.md itself is unchanged. Both files are hashed in
results/preregistration_sha256.txt, and optc_budget.py checks both hashes.

The hypotheses H1–H3, the scores, a = 1e5, the units, the budgets and the labels stay as frozen, except where an item
below says otherwise.

## A1. First-contact count k
- k is the number of **distinct destinations** in the unit that the source had never contacted before. This is how
  the code (hostunit.py), the LANL development analysis and the paper all define it.
- It replaces the protocol's wording "the number of those [flows] whose destination the source has never contacted".
- D is the source's earlier distinct destinations and N its earlier flows, as frozen.

## A2. Powell-style window
- The roles are fitted on the previous **168 observed hourly units** for the hour unit, and on the previous
  **7 observed days** for the day unit.
- "Observed" means the units are counted over the data actually present, so the window skips the 20–22 Sep gap that
  is not downloaded.

## A3. Labels and deduplication
- **Primary:** a five-tuple's records all carry the largest label among them, so that removing the two-ended
  duplicate cannot drop a label.
- **Sensitivity:** the same results are also reported with LMDEval's exact rule, where a label stays only on the
  record that is kept.

## A4. H3 source set
"Labelled lateral-movement source hosts that have logs" means the labelled sources that appear as a reporting
hostname in the filtered OpTC records. It is no longer defined as every source except names starting with DC1.

## A5. Missing and infinite scores
- A missing score (NaN), such as a unit with no Powell score, ranks lowest.
- An infinite E ranks highest, with ties broken at random as for any other tie.

## A6. Timing of alerts, delay and coverage
- Under the daily top-b rule, an alert on any unit of day t becomes available at the end of day t.
- Delay is reported in days.
- Coverage is the share of labelled events in days **after** the alert day, which is the part an analyst could have
  interrupted. This replaces "on or after".

## A7. Additional secondary analyses
These were declared now, before any evaluation data were read.
1. **History split.** Recall and sources are reported separately for positive units whose source has earlier history
   (N > 0) and for those whose source does not.
2. **Stronger novelty control:** first contacts of new hosts, scored k if N = 0 and 0 otherwise.
3. **Precision** at each b.
4. **Threshold rule:** both the mean and the median number of alerts per day.

## A8. External timestamp
Both protocol files and the analysis code are to be published as a tagged commit before the evaluation is run, if the
authors approve. The commit time and the tag then serve as the external timestamp.
