# OpTC held-out evaluation: frozen protocol

Written 2026-10-29, before any OpTC flow record was read. The detector and every analysis choice below were fixed on
the LANL 2015 development data (practical/results/lanl_budget.json). The file's SHA-256 is recorded in
practical/results/preregistration_sha256.txt, and the OpTC scripts check it before they write results.
Anything not fixed here is reported as exploratory.

## Data
- DARPA OpTC eCAR, download option A:
  - evaluation/23Sep19-red, evaluation/24Sep19, evaluation/25Sept;
  - benign/19Sep19, which gives the hosts history before the exercise.
  - 124 files, 218.1 GB compressed, listed in optc/optc_files_optionA.tsv.
- Events: FLOW START with both addresses internal. The selection, the IP-to-host mapping (optc_known_addresses.json plus
  mappings learnt from outbound records) and the two-ended deduplication follow LMDEval's extract_optc.py
  (Larroche et al., commit 2b95a68).
- Flows whose endpoint cannot be mapped to a host name keep the IP address as their identity.
- Time is local exercise time (UTC-4), and day boundaries are local midnight.
- Labels: LMDEval optc_redteam.csv.
  - **Primary:** "Lateral movement" ids only (249).
  - **Secondary:** all red-team ids (1,359).
- A source host is a labelled attack source in a unit if at least one of its flows in that unit carries the label.
  For flows seen only at the destination (DC1 has no logs), the source is the flow's source address.

## Unit and budget
- **Primary unit:** source host × hour. OpTC has about 500 logging hosts, so host-hours per day (~12,000) match LANL's
  host-days per day (~11,000).
- **Budget:** each day the analyst receives that day's b top-scoring host-hours, for b = 1, 10 and 100. Ties are
  broken at random, and we report the exact expectation.
- **Secondary unit:** source host × day, with b = 1 and 10.
- **Evaluation period:** every hour of 23, 24 and 25 Sep. 25 Sep has no internal lateral movement, so it contributes
  only negatives.
- **History:** all earlier units in the downloaded data, beginning with benign/19Sep19. The gap between 19 and 23 Sep
  is not downloaded.

## Scores (higher = more suspicious)
- **E, the primary detector:** first-contact binomial surprise.
  - Score: E = -ln P(Binomial(n, q) >= k), where n is the unit's flows from the source and k is the number of those
    whose destination the source has never contacted before.
  - Rate: q = (D + a·q_pop)/(N + a), where D and N are the source's earlier distinct destinations and flows, and
    q_pop = (new_total + 1)/(flows_total + 2) over all earlier units of the network.
  - The prior weight a = 1e5 was chosen on LANL by the rule "largest mean recall over b = 1, 10, 100; ties go to the
    smaller a". a = ∞ is reported as a sensitivity check.
- X_host: cross-entropy against the host's own history, Laplace α = 1, with K = the number of distinct host
  identities in the data.
- KL_host = X_host − H_today.
- First-contact count (new_dst), the new-edge baseline.
- Novelty baselines: host age (−hours since the source was first seen) and −log(1 + earlier flows).
- Powell-style role baseline (Powell 2020, adapted from per-user to network-wide graphs because OpTC flows carry no
  user):
  - Each unit's directed host graph gives every vertex these features: log1p of out-degree, in-degree, out-flows and
    in-flows; local clustering coefficient; PageRank.
  - Features are min–max scaled on the history window.
  - A rank-5 NMF is fitted on the previous 7 days of units (causal). The score is the reconstruction error of the
    source vertex.

## Metrics
- Recall@b of positive host-units, with a 95% interval from a bootstrap that resamples source hosts (2,000 draws).
- Precision@b.
- Source hosts detected: at least one positive unit alerted with probability ≥ 0.5.
- Delay from a source's first positive unit to its first alert.
- Share of labelled events at or after their source's first alert.
- Within-day AUC, reported for comparison only.

## Pass / fail statements, fixed in advance
- **H1:** E has a higher recall@10 (primary unit and labels) than X_host.
- **H2:** E has a higher recall@10 than the first-contact count.
- **H3:** E detects at least half of the labelled lateral-movement source hosts that have logs at b = 10.
- Each is reported as supported or not, whatever the outcome. With a handful of source hosts, none is a significance
  test.
