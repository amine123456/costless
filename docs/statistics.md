# How gating works

LLM applications are non-deterministic: the same prompt and input can produce a
passing answer on one run and a failing one on the next. A gate that compares
two single runs would block merges on noise and miss real regressions that
happen to hide behind it. costless only blocks when the evidence says a change
made things worse, and worse by more than you have said you care about.

## 1. Measure each case several times

Every case is run `run.repeats` times (default 5). For each attempt costless
records the following:

| Value | Definition |
|---|---|
| quality | weighted mean of the scorer scores, in [0, 1] |
| failed | 1 if the attempt errored or any scorer failed, else 0 |
| cost | priced token usage of the system under test (judges excluded) |
| latency | wall-clock time of the target call |

Repeats estimate how much each case varies by itself. A case that passes 3 times
out of 5 is flagged `flaky` in the results.

## 2. Compare paired cases

The baseline (latest run on the main branch) and the candidate (this merge
request) are compared **case by case**. Only cases present in both runs are used,
and cases that were added or removed are listed in the report. Pairing removes
most of the variation between easy and hard cases, which is usually much larger
than the effect of a change.

The metrics are the same ones shown in the run summary:

| Metric | Statistic | Compared as |
|---|---|---|
| Quality | mean over cases of the per-case mean quality | difference |
| Failure rate | mean over cases of the per-case failure share | difference |
| Cost per case | mean over cases of the per-case mean cost | ratio |
| p95 latency | 95th percentile over all attempts | ratio |

## 3. Paired cluster bootstrap

To get a confidence interval for each difference or ratio, costless uses a
**paired cluster bootstrap** with 10,000 resamples by default.

1. Draw `n` cases with replacement from the `n` paired cases. Both runs use
   the same draw, which keeps the pairing.
2. For each run, recompute the statistic over all attempts of the drawn cases.
3. Record the candidate minus baseline (or the ratio of the two).

The 2.5th and 97.5th percentiles of the 10,000 recorded values form the 95%
interval. Resampling cases captures both kinds of uncertainty:

- **Which cases happen to be in the dataset.**
- **Which answers the model happened to give.** That variation is already
  contained in each case's attempts.

The bootstrap needs no normality assumption, and it handles bounded scores,
pass/fail data and skewed latencies in the same way. The generator is seeded
(`gate.seed`), so the same two runs always give the same report.

**Rejected: also resampling the repeats.** We first resampled the repeats inside
every case as well, which is a two-stage bootstrap. On simulated runs this
produced intervals that were too wide: the within-case noise ends up counted
twice, once inside the case means and once in the second resampling stage. It
caught a real 15-point drop in only about 60% of trials. Resampling the cases
alone matches a textbook paired t-test, as the next table shows.

## 4. Decision rule

Each metric gets a tolerance in `costless.yaml`: an allowed drop in quality,
an allowed rise in failure rate, and an allowed percentage rise in cost or p95
latency. Each metric then ends up in one of these states:

| Status | Condition | Blocks the merge |
|---|---|---|
| **regressed** | the whole interval is on the "worse" side of no change, *and* the observed change is worse than the tolerance | yes |
| **inconclusive** | the observed change is worse than the tolerance, but the interval still includes no change | no (yes with `fail_on_inconclusive`) |
| **improved** | the whole interval is on the "better" side | no |
| **ok** | anything else | no |
| **breached** | an absolute limit (`min`, `max`, `max_ms`) is violated by this run alone | yes |
| **unavailable** | cannot be computed, for example because a model has no price | no |

The rule combines two conditions, and a merge is blocked only when both hold:

- **The change is real:** statistically significant at the chosen confidence level.
- **It matters:** larger than the tolerance you set.

An *inconclusive* result means the data cannot yet tell; the remedy is more
repeats or more cases.

Without a baseline (the very first run on the main branch) only absolute
limits are checked.

## 5. Measured properties

The test suite simulates pairs of runs to check the method:

- 30 cases with different difficulties (pass probability drawn from Beta(4, 1.5))
- 5 repeats per case
- tolerance 0.02

Each row was measured over 300 simulated comparisons (`tests/test_compare.py`
asserts bounds on smaller samples):

| Scenario | Paired t-test (reference) | costless bootstrap |
|---|---|---|
| No change: share of runs that would block (false alarms) | 3% | 4% |
| 15-point drop in pass probability: share detected | 85% | 87% |
| 20-point drop: share detected | 96% | 97% |

These figures come from simulated data, not from a real LLM. They show that the
procedure behaves as designed; they say nothing about any particular model.

## 6. Choosing repeats, cases and tolerances

The smallest drop the gate can reliably detect is roughly

```
3 × sd(per-case difference) / sqrt(number of cases)
```

For pass/fail scores with 5 repeats, the per-case difference typically has a
standard deviation around 0.25 to 0.3. That makes about 0.15 detectable with 30
cases and about 0.1 with 70 cases. The ways to make the gate more sensitive are:

- **More cases.** Most effective, since it divides by `sqrt(n)`.
- **More repeats.** This reduces the within-case part of the noise, with
  diminishing returns past about 5 repeats.
- **Continuous scores** (an LLM judge on a 1-5 scale), which carry more
  information than pass/fail.

Set tolerances to the smallest change you would actually act on. A tolerance
below what the dataset can detect only produces *inconclusive* results.

## 7. Limitations

- **Several metrics are tested.** Each has about a 2.5% one-sided chance of a
  false alarm. With four metrics, the chance that any of them blocks a no-op
  change can reach about 10% in the worst case. Raise `gate.confidence`, for
  example to 0.99, if false alarms are costly.
- **Cases are assumed independent.** Near-duplicate cases make the interval
  too narrow.
- **Very small datasets (fewer than about 10 cases) give unreliable intervals.**
  The bootstrap needs enough cases to resample from.
- **Latency depends on the machine and the time of day.** Baseline and
  candidate run at different times, often on different CI runners. Use a
  generous latency tolerance, or `max_ms` as an absolute limit.
- **A changed dataset changes what is being compared.** The report flags it,
  and only common cases are compared.
- **LLM-judge scores carry the judge's own noise and bias.** Calibrate the
  judge (`costless calibrate`) before gating on it.
